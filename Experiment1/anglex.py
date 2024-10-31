
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap
from jax_md import util
from jax_md import partition
from jax_md import dataclasses
from jax_md import quantity
from jax_md import space
from functools import partial
from typing import Dict, Callable, List, Tuple, Union, Optional

high_precision_sum = util.high_precision_sum
# Typing

Array = util.Array
PyTree = util.PyTree

f32 = util.f32
f64 = util.f64

i32 = util.i32
i64 = util.i64

DisplacementOrMetricFn = space.DisplacementOrMetricFn

#print("start")

def angx(fn: Callable[..., Array],
         displacement_or_metric: DisplacementOrMetricFn,
         static_angs: Optional[Array]=None,
         static_bond_types: Optional[Array]=None,
         ignore_unused_parameters: bool=False,
         **kwargs) -> Callable[..., Array]:

  merge_dicts = partial(util.merge_dicts,
                        ignore_unused_parameters=ignore_unused_parameters)

  def ang_rad(v1,v2):
    #return jnp.dot(v1, v2)
    return jnp.rad2deg(jnp.arccos(jnp.dot(v1, v2)/(jnp.linalg.norm(v1)*jnp.linalg.norm(v2))))

  def compute_fn(R, angs,  static_kwargs, dynamic_kwargs):
    Ra = R[angs[:, 0]]
    Rb = R[angs[:, 1]]
    Rc = R[angs[:, 2]]

    d = vmap(displacement_or_metric)
    ba = d(Ra,Rb)
    bc = d(Rc,Rb)
    # print("ba:")
    # print(ba)
    # print("bc:")
    # print(bc)

    ar=vmap(ang_rad)
    ct = ar(ba,bc)
    # print("ct:")
    # print(ct)
    # print("cted")
    _kwargs = merge_dicts(static_kwargs, dynamic_kwargs)

    return high_precision_sum(fn(ct, **_kwargs))

  def mapped_fn(R: Array,
                angs: Optional[Array]=None,
                **dynamic_kwargs) -> Array:
    accum = f32(0)
    return accum + compute_fn(R, static_angs, kwargs, dynamic_kwargs)
  return ang_rad,mapped_fn

#print("test")

def setup_periodic_box():
  def displacement_fn(Ra, Rb, **unused_kwargs):
    dR = Ra - Rb
    return dR
    #return np.mod(dR + box_size * f32(0.5), box_size) - f32(0.5) * box_size

  def shift_fn(R, dR, **unused_kwargs):
    return R+dR
    #return np.mod(R + dR, box_size)

  return displacement_fn, shift_fn

displacement, shift = setup_periodic_box()



def fs(s,s0=60.0,k=0.2):
  return k*(s-s0)**2

R=jnp.array([[1.0,1.0,0.0],[0.0,0.0,0.0],[1.0,1.0,1.0],[0.0,1.0,2.0]])
a=jnp.array([[0,1,2],[0,1,3]])
#print(R)
#print(a)
# print(displacement(R[0],R[1]))
# ds=space.canonicalize_displacement_or_metric(displacement)
# print(ds(R[0],R[1]))

#f1,f2=angx(fs,space.canonicalize_displacement_or_metric(displacement),a)
f1,f2=angx(fs,displacement,a)
#print(f2(R))
#print(f1(R[0]-R[1],R[2]-R[1]))

#print("end")
