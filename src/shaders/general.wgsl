override workgroup_size_x: u32 = 1u;
override workgroup_size_y: u32 = 1u;
override workgroup_size_z: u32 = 1u;
override dispatch_count_x: u32 = 1u;
override dispatch_count_y: u32 = 1u;
override dispatch_count_z: u32 = 1u;

// var<workgroup> workgroup_size : array<u32, wgsize * 2>;


struct Scalar {
    density: f32,
}


struct ScalarField {
    field: array<Scalar>
}

struct Mesh {
    vertices: array<vec3<u32>>,
}

@group(0) @binding(0)
var<storage> in: ScalarField;

@group(0) @binding(1)
var<storage, read_write> out: Mesh;




// Max: 256, 256, 64, Max(x*y*z) = 256 = 2^8

@compute @workgroup_size(8,8,4)
fn main(
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id : vec3<u32>,
    @builtin(global_invocation_id) global_invocation_id : vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
      // workgroup_index is similar to local_invocation_index except for
  // workgroups, not threads inside a workgroup.
  // It is not a builtin so we compute it ourselves.

  let threads_per_workgroup = workgroup_size_x * workgroup_size_y * workgroup_size_z;

 
  let workgroup_index =  
     workgroup_id.x +
     workgroup_id.y * num_workgroups.x +
     workgroup_id.z * num_workgroups.x * num_workgroups.y;
 
  // global_invocation_index is like local_invocation_index
  // except linear across all invocations across all dispatched
  // workgroups. It is not a builtin so we compute it ourselves.
 
  let global_invocation_index =
     workgroup_index * threads_per_workgroup +
     local_invocation_index;
 
     out.vertices[global_invocation_index] = global_invocation_id;
   




}