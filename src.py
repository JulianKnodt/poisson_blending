import numpy as np
from scipy.sparse import linalg
from scipy.sparse import lil_matrix

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

  in_mask = mask[u,v] == 1
  # computing A
  m = mask[clamp_u(u+1),v] == 1
  A[m & in_mask] = -1
  m = mask[clamp_u(u-1),v] == 1
  A[m & in_mask] = -1
  m = mask[u,clamp_v(v+1)] == 1
  A[m & in_mask] = -1
  m = mask[u,clamp_v(v-1)] == 1
  A[m & in_mask] = -1
  A = A.tocsr()

  # computing b
  b = np.zeros((N,C))
  b[range(N)] = lapl(src, idxs)
  locs = loc(mask, idxs)
  on_edge = locs == ON_EDGE
  # squeeze last
  sq_l = lambda v: v[..., None]
  b = np.where(
    sq_l(on_edge & (mask[clamp_u(u+1),v] == 0)),
    dst[clamp_u(u+1),v],
    b,
  )
  b = np.where(
    sq_l(on_edge & (mask[np.maximum(u-1,0),v] == 0)),
    dst[np.minimum(u+1,mask.shape[0]-1),v],
    b,
  )
  b = np.where(
    sq_l(on_edge & (mask[u,np.minimum(v+1,mask.shape[1]-1)] == 0)),
    dst[u,np.minimum(v+1,mask.shape[1]-1)],
    b,
  )

  # done computing b
  x = np.stack([
    linalg.cg(A, b_c)[0]
    #linalg.spsolve(A, b_c)
    for b_c in np.split(b, b.shape[1], axis=1)
  ], axis=1)
  composite = np.copy(dst)
  composite[u,v] = x
  return composite

N = 128
out = poisson_blend(
  src=np.random.randn(N,N,3),
  dst=np.random.randn(N,N,3),

  mask=(np.random.rand(N,N) - 0.4).round()
)
