use std::collections::HashMap;

use encase::{ArrayLength, ShaderType, StorageBuffer};
use futures::executor::block_on;
use mint::{Vector2, Vector3, Vector4};
use wgpu::{include_wgsl, util::DeviceExt, BufferAddress};

#[derive(Debug, ShaderType, PartialEq)]
struct Scalar {
    density: f32,
}

#[derive(Debug, ShaderType, PartialEq)]
struct ScalarField {
    #[size(runtime)]
    field: Vec<Scalar>,
}

#[derive(Debug, ShaderType, PartialEq)]
struct Mesh {
    #[size(runtime)]
    vertices: Vec<Vector3<u32>>,
}

fn run<IN: encase::ShaderType, OUT: encase::ShaderType>(
    shader: wgpu::ShaderModuleDescriptor,
    data: &[u8],
    is_uniform: bool,
) -> Vec<u8> {
    //2^27 is the maximum buffer size. Number of elements is 2^23. Size of element is: 4 (2^2) bytes

    const workgroup_size: (u64, u64, u64) = (2_u64.pow(3), 2_u64.pow(3), 2_u64.pow(2));
    // const dispatch_count: (u64, u64, u64) = (65535, 1, 1);
    const dispatch_count: (u64, u64, u64) = (2_u64.pow(5), 2_u64.pow(5), 2_u64.pow(5));

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
        ..Default::default()
    });
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        ..Default::default()
    }))
    .unwrap();

    println!("Adapter info: {:#?}", adapter.get_info());

    let (device, queue) =
        block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).unwrap();

    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
    });

    let output_gpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: OUT::min_size().get()
            * workgroup_size.0
            * workgroup_size.1
            * workgroup_size.2
            * dispatch_count.0
            * dispatch_count.1
            * dispatch_count.2,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mapping_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mapping Buffer"),
        size: OUT::min_size().get()
            * workgroup_size.0
            * workgroup_size.1
            * workgroup_size.2
            * dispatch_count.0
            * dispatch_count.1
            * dispatch_count.2,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(IN::min_size()),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(OUT::min_size()),
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let shader = device.create_shader_module(shader);

    let limits = wgpu::Limits::default();

    let constants = HashMap::<String, f64>::from([
        ("workgroup_size_x".into(), workgroup_size.0 as f64),
        ("workgroup_size_y".into(), workgroup_size.1 as f64),
        ("workgroup_size_z".into(), workgroup_size.2 as f64),
        ("dispatch_count_x".into(), dispatch_count.0 as f64),
        ("dispatch_count_y".into(), dispatch_count.1 as f64),
        ("dispatch_count_z".into(), dispatch_count.2 as f64),
    ]);

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            zero_initialize_workgroup_memory: true,
            vertex_pulling_transform: false,
        },
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_gpu_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(
            dispatch_count.0 as u32,
            dispatch_count.1 as u32,
            dispatch_count.2 as u32,
        );
    }

    let _copy_size: BufferAddress = output_gpu_buffer.size();

    encoder.copy_buffer_to_buffer(
        &output_gpu_buffer,
        0,
        &mapping_buffer,
        0,
        OUT::min_size().get()
            * workgroup_size.0
            * workgroup_size.1
            * workgroup_size.2
            * dispatch_count.0
            * dispatch_count.1
            * dispatch_count.2,
    );

    queue.submit(core::iter::once(encoder.finish()));

    let output_slice = mapping_buffer.slice(..);
    output_slice.map_async(wgpu::MapMode::Read, |_| {});

    device.poll(wgpu::Maintain::Wait);

    let output = output_slice.get_mapped_range().to_vec();

    mapping_buffer.unmap();
    output
}

fn main() {
    let scalar_field = ScalarField {
        field: Vec::from([Scalar { density: 1. }]),
    };

    let mut in_byte_buffer = Vec::new();
    let mut in_buffer = StorageBuffer::new(&mut in_byte_buffer);
    in_buffer.write(&scalar_field).unwrap();

    let shader = include_wgsl!("./shaders/general.wgsl");

    let out_byte_buffer = run::<ScalarField, Mesh>(shader, &in_byte_buffer, false);

    let out_buffer = StorageBuffer::new(out_byte_buffer);
    let mesh: Mesh = out_buffer.create().unwrap();

    println!("{:?}", &mesh.vertices.last());
}
