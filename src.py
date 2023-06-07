import numpy as np
from scipy.sparse import linalg
from scipy.sparse import lil_matrix

def adj(v):
  i,j = v
  return [
    (i+1,j),
    (i-1,j),
    (i,j+1),
    (i,j-1),
  ]

# Apply the Laplacian operator at a given index
def lapl(source, idxs):
  u = idxs[:, 0]
  v = idxs[:, 1]
  val = 4 * source[u, v]

  m = u + 1 < source.shape[0]
  val[m] -= source[(u+1)[m],v[m]]

  m = u > 0
  val[m] -= source[(u-1)[m],v[m]]

  m = v + 1 < source.shape[1]
  val[m] -= source[u[m],(v+1)[m]]

  m = v > 0
  val[m] -= source[u[m],(v-1)[m]]

  return val

ON_EDGE = 2

def loc(mask, idxs):
  u = idxs[:, 0]
  v = idxs[:, 1]

  on_edge = (mask[np.minimum(u+1,mask.shape[0]-1),v] == 0) | \
    (mask[np.maximum(u-1,0),v] == 0) | \
    (mask[u,np.minimum(v+1,mask.shape[1]-1)] == 0) | \
    (mask[u,np.maximum(v-1,0)] == 0)

  return np.where(
    mask[u,v] == 0, 0,
    # otherwise in mask, could be on edge
    np.where(on_edge, ON_EDGE, 1)
  )

def poisson_blend(
  src: ["N","N","C"],
  dst: ["N","N","C"],
  mask: ["N","N"],
):
  idxs = np.stack(np.nonzero(mask), axis=1)
  assert(idxs.shape[1] == 2), idxs.shape
  u = idxs[:, 0]
  v = idxs[:, 1]

  N = idxs.shape[0]
  C = src.shape[-1]

  A = lil_matrix((N,N))
  A[range(N), range(N)] = 4

  clamp_u = lambda v: np.clip(v, 0, src.shape[0]-1)
  clamp_v = lambda v: np.clip(v, 0, src.shape[1]-1)

  pts = { tuple(x[0].tolist()): i for i,x in enumerate(np.split(idxs, N, axis=0)) }
  for p,i in pts.items():
    for a in adj(p):
      if a not in pts: continue
      j = pts[a]
      A[i,j] = -1

  A = A.tocsr()

  # computing b
  b = np.zeros((N,C))
  b[range(N)] = lapl(src, idxs)
  locs = loc(mask, idxs)
  for pt,i in pts.items():
    if locs[i] != ON_EDGE: continue
    for a in adj(pt):
      if mask[a[0], a[1]] == 0:
        b[i] += dst[a[0], a[1]]

  # done computing b
  x = np.stack([
    linalg.cg(A, b_c)[0]
    #linalg.spsolve(A, b_c)
    for b_c in np.split(b, b.shape[1], axis=1)
  ], axis=1)
  composite = np.copy(dst)
  composite[u,v] = x
  return composite

def main():
  import os
  import cv2
  src = cv2.imread("tmp/source.png")/255.
  dst = cv2.imread("tmp/target.png")/255.
  mask = cv2.imread("tmp/mask.png")/255.
  mask[mask != 1] = 0
  mask = mask[...,2]
  out = poisson_blend(src=src, dst=dst, mask=mask)
  print(out.min(), out.max())
  cv2.imwrite("out.png", out * 255.)

if __name__ == "__main__": main()

