import torch
import torch.nn.functional as F




# https://github.com/pytorch/pytorch/issues/50334
def interp(x, xp, fp):
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    assert(xp.shape[-1] == fp.shape[-1])
    m = (fp[..., 1:] - fp[..., :-1]) / (xp[..., 1:] - xp[..., :-1])
    b = fp[..., :-1] - (m * xp[..., :-1])

    idxs = x[..., :, None] >= xp[..., None, :]
    idxs = idxs.sum(dim=-1) - 1
    idxs = idxs.clamp(0, m.shape[-1] - 1)

    return m[..., idxs] * x + b[..., idxs]


def histo_match(
  src,
  dst,
):
  assert(len(src.shape) == 3)
  assert(len(dst.shape) == 3)
  N = src.shape[0]
  C = src.shape[1]
  assert(src.shape[1] == dst.shape[1])
  chans = []
  for sc, dc in zip(src.split(1, dim=1), dst.split(1, dim=1)):

    src_uniq,src_cnts = sc.unique(return_counts=True,dim=-1)

    dst_uniq,dst_inv_idx,dst_cnts = dc.unique(return_inverse=True,return_counts=True,dim=-1)

    dst_quantiles = torch.cumsum(dst_cnts, dim=-1)
    dst_cdf = dst_quantiles/dst_quantiles[..., -1]

    src_quantiles = torch.cumsum(src_cnts, dim=-1)
    src_cdf = src_quantiles/src_quantiles[..., -1]

    interp_values = interp(dst_cdf[None, None], src_cdf[None, None], src_uniq)

    chans.append(interp_values[..., dst_inv_idx])
  return torch.cat(chans, dim=1)

def histo_match_images(src, dst):
  return histo_match(src.flatten(-2), dst.flatten(-2)).reshape_as(dst)

if __name__ == "__main__":
  def arguments():
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--src", type=str)
    a.add_argument("--dst", type=str)
    a.add_argument("--out", type=str, default="out.png")
    return a.parse_args()
  def main():
    args = arguments()
    import torchvision as tv
    src = tv.io.read_image(args.src)/255.
    dst = tv.io.read_image(args.dst)/255.
    sz = 512
    dst = dst[None]
    dst = F.interpolate(dst, size=(sz, sz))

    src = src[None]
    src = F.interpolate(src, size=(sz, sz))
    out = histo_match_images(src, dst)
    tv.utils.save_image(out, args.out)
    from skimage.exposure import match_histograms
    out = match_histograms(dst.squeeze(0).numpy(), src.squeeze(0).numpy(), channel_axis=0)
    out = torch.from_numpy(out)
    tv.utils.save_image(out[None], "skimg.png")
  main()
