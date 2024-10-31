
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
         static_r: Optional[Array]=None,
         static_bond_types: Optional[Array]=None,
         ignore_unused_parameters: bool=False,
         **kwargs) -> Callable[..., Array]:

  merge_dicts = partial(util.merge_dicts,
                        ignore_unused_parameters=ignore_unused_parameters)


  def compute_fn(R, R0,  static_kwargs, dynamic_kwargs):

    d = vmap(displacement_or_metric)
    dr = d(R,R0)
    
    #print("dr:",dr)

    _kwargs = merge_dicts(static_kwargs, dynamic_kwargs)

    return high_precision_sum(fn(dr, **_kwargs))

  def mapped_fn(R: Array,
                angs: Optional[Array]=None,
                **dynamic_kwargs) -> Array:
    accum = f32(0)
    return accum + compute_fn(R, static_r, kwargs, dynamic_kwargs)
  return mapped_fn

# print("test")

# def setup_periodic_box():
#   def displacement_fn(Ra, Rb, **unused_kwargs):
#     dR = Ra - Rb
#     return dR
#     #return np.mod(dR + box_size * f32(0.5), box_size) - f32(0.5) * box_size

#   def shift_fn(R, dR, **unused_kwargs):
#     return R+dR
#     #return np.mod(R + dR, box_size)

#   return displacement_fn, shift_fn

# displacement, shift = setup_periodic_box()

# dis=space.canonicalize_displacement_or_metric(displacement)

# def fs(s,s0=60.0,k=0.2):
#   return k*s0*s

# R1=jnp.array([[1.0,1.0],[0.0,0.0],[1.0,2.0],[0.0,1.0]])
# R2=jnp.array([[1.0,1.0],[3.0,4.0],[1.0,2.0],[0.0,1.0]])

# print(R1)
# print(R2)

# f1=angx(fs,dis,R1)
# print(f1(R2))
