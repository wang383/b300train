"""
测量 SDXL InstructPix2Pix 训练单步各环节耗时
使用已训练的 checkpoint（支持8通道输入）
"""
import os, time, torch, torch.distributed as dist
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

dist.init_process_group("nccl")
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
device = torch.device(f"cuda:{rank}")
is_main = rank == 0

def ts(): torch.cuda.synchronize(); return time.perf_counter()

if is_main: print("加载模型（使用已训练checkpoint）...", flush=True)

model_id  = "stabilityai/stable-diffusion-xl-base-1.0"
ckpt_dir  = "/workspace/checkpoints"

# UNet 从 checkpoint 加载（已支持8通道）
unet = UNet2DConditionModel.from_pretrained(ckpt_dir, subfolder="unet", torch_dtype=torch.bfloat16).to(device)
vae  = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16).to(device)
text_enc1 = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
text_enc2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16).to(device)
tok1 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
tok2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[rank])
optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-6)

if is_main: print("加载数据集...", flush=True)

dataset = load_dataset("fusing/instructpix2pix-1000-samples", split="train")
tf = transforms.Compose([
    transforms.Resize(1024),
    transforms.CenterCrop(1024),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def collate(batch):
    imgs    = torch.stack([tf(b["input_image"].convert("RGB")) for b in batch])
    edited  = torch.stack([tf(b["edited_image"].convert("RGB")) for b in batch])
    prompts = [b["edit_prompt"] for b in batch]
    return {"pixel_values": imgs, "edited_pixel_values": edited, "prompts": prompts}

sampler = torch.utils.data.distributed.DistributedSampler(dataset)
loader  = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=4,
                     pin_memory=True, collate_fn=collate)

if is_main: print("\n开始 profiling（5 steps）...\n", flush=True)

keys = ["data_load","vae_encode","text_encode","noise_add","unet_forward","backward","nccl_sync","optimizer"]
timings = {k: [] for k in keys}

loader_iter = iter(loader)
for step in range(5):
    t0 = ts()
    batch  = next(loader_iter)
    imgs   = batch["pixel_values"].to(device, dtype=torch.bfloat16, non_blocking=True)
    edited = batch["edited_pixel_values"].to(device, dtype=torch.bfloat16, non_blocking=True)
    torch.cuda.synchronize()
    timings["data_load"].append(ts()-t0)

    t0 = ts()
    with torch.no_grad():
        latents      = vae.encode(edited).latent_dist.sample() * vae.config.scaling_factor
        orig_latents = vae.encode(imgs).latent_dist.sample()   * vae.config.scaling_factor
    torch.cuda.synchronize()
    timings["vae_encode"].append(ts()-t0)

    t0 = ts()
    prompts = batch["prompts"]
    with torch.no_grad():
        t1 = tok1(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        t2 = tok2(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        e1 = text_enc1(**t1, output_hidden_states=True)
        e2 = text_enc2(**t2, output_hidden_states=True)
        prompt_embeds = torch.cat([e1.hidden_states[-2], e2.hidden_states[-2]], dim=-1)
        pooled_embeds = e2.text_embeds
    torch.cuda.synchronize()
    timings["text_encode"].append(ts()-t0)

    t0 = ts()
    noise     = torch.randn_like(latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
    noisy     = noise_scheduler.add_noise(latents, noise, timesteps)
    concat    = torch.cat([noisy, orig_latents], dim=1)
    add_time_ids = torch.tensor([[1024,1024,0,0,1024,1024]], device=device).repeat(latents.shape[0],1)
    torch.cuda.synchronize()
    timings["noise_add"].append(ts()-t0)

    t0 = ts()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = unet(concat, timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids}).sample
        loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
    torch.cuda.synchronize()
    timings["unet_forward"].append(ts()-t0)

    t0 = ts()
    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()
    timings["backward"].append(ts()-t0)

    t0 = ts()
    dist.barrier()
    torch.cuda.synchronize()
    timings["nccl_sync"].append(ts()-t0)

    t0 = ts()
    optimizer.step()
    torch.cuda.synchronize()
    timings["optimizer"].append(ts()-t0)

    if is_main:
        total = sum(timings[k][-1] for k in keys)
        print(f"Step {step+1}: {total*1000:.0f}ms | " +
              " | ".join(f"{k}={timings[k][-1]*1000:.0f}ms" for k in keys), flush=True)

if is_main:
    print("\n========== 平均耗时 ==========")
    total_avg = sum(sum(v)/len(v) for v in timings.values())
    for k in keys:
        avg = sum(timings[k])/len(timings[k])*1000
        print(f"  {k:<15}: {avg:7.1f} ms  ({avg/total_avg*100:.1f}%)")
    print(f"  {'总计':<15}: {total_avg*1000:7.1f} ms")

dist.destroy_process_group()
