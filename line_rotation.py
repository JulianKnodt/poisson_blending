import torch
import torchvision as tv
import torch.nn.functional as F

def nearest_sample(img, uv):
  w,h = img.shape[1:3]
  assert(w == h)
  s = w
  assert(uv.shape[-1] == 2)
  u,v = [(val * s).round().long() for val in [uv[..., 0], uv[..., 1]]]
  return img[:,v,u]

def nearest_pix_put(img, uv, vals):
  w,h = img.shape[1:3]
  assert(w == h)
  s = w
  assert(uv.shape[-1] == 2)
  u,v = [(val * s).round().long() for val in [uv[..., 0], uv[..., 1]]]
  img[:,v,u] = vals

def line_rotate(img, s0, e0, s1, e1):
  w,h = img.shape[1:3]
  assert(w == h)
  s = w

  l0 = (e0 - s0)
  l1 = (e1 - s1)

  l1_pix = int((l0 * s).norm(dim=-1).ceil().item())

  l0_dir = F.normalize(l0, dim=-1)
  l1_dir = F.normalize(l1, dim=-1)
  # have to do inverse sampling from l1 dst to l0 src
  rot = rotation_btwn(l1_dir, l0_dir)
  #way to perform rotation is: (rot * l1_dir[None, :]).sum(dim=-1)
  t = torch.linspace(0, 1, steps= l1_pix)
  samples = t[:, None] * l1[None, :]
  # rotate to s0
  rot_samples = (rot[None] * samples[..., None, :]).sum(dim=-1)
  sampled_pix = nearest_sample(img, rot_samples + s0[None, :])
  nearest_pix_put(img, samples + s1[None, :], sampled_pix)
  return img


def rotation_btwn(src, tgt):
  assert(src.shape[-1] == 2)
  assert(tgt.shape[-1] == 2)
  sx, sy = src.split([1,1], dim=-1)
  tx, ty = tgt.split([1,1], dim=-1)
  r0 = torch.cat([sx * tx + sy * ty, tx * sy - sx * ty], dim=-1)
  r1 = torch.cat([sx * ty - tx * sy, sx * tx + sy * ty], dim=-1)
  return torch.stack([r0, r1], dim=-2)

if __name__ == "__main__":
  s0 = torch.tensor([0.25, 0.25])
  e0 = torch.tensor([0.25, 0.75])

  s1 = torch.tensor([0.625, 0.75])
  e1 = torch.tensor([0.875, 0.25])

  # intentionally swapped
  e2 = torch.tensor([0.65, 0.75])
  s2 = torch.tensor([0.9, 0.25])

  img = torch.zeros(1, 256, 256, 3)
  img[:, :128, 128:, 0] = 1
  img[:, 128:, :128, 1] = 1
  img[:, 128:, 128:, 2] = 1
  line_rotate(img, s0, e0, s1, e1)
  line_rotate(img, s0, e0, s2, e2)
  tv.utils.save_image(img.movedim(-1, 1), "tmp.png")
