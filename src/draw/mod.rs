use bytemuck::{Pod, Zeroable};

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
}

impl Vertex for ModelVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

pub fn sample_vertices() -> Vec<ModelVertex> {
    vec![
        ModelVertex {
            position: [-0.0868241, 0.49240386, 0.0],
            tex_coords: [0.4131759, 0.00759614],
            normal: [0.0, 0.0, 0.0],
        }, // A
        ModelVertex {
            position: [-0.49513406, 0.06958647, 0.0],
            tex_coords: [0.0048659444, 0.43041354],
            normal: [0.0, 0.0, 0.0],
        }, // B
        ModelVertex {
            position: [-0.21918549, -0.44939706, 0.0],
            tex_coords: [0.28081453, 0.949397],
            normal: [0.0, 0.0, 0.0],
        }, // C
        ModelVertex {
            position: [0.35966998, -0.3473291, 0.0],
            tex_coords: [0.85967, 0.84732914],
            normal: [0.0, 0.0, 0.0],
        }, // D
        ModelVertex {
            position: [0.44147372, 0.2347359, 0.0],
            tex_coords: [0.9414737, 0.2652641],
            normal: [0.0, 0.0, 0.0],
        }, // E
    ]
}

pub fn sample_indices() -> Vec<u16> {
    vec![0, 1, 4, 1, 2, 4, 2, 3, 4]
}
