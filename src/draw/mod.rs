use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    pub fn new(position: [f32; 3], color: [f32; 3]) -> Self {
        return Self { position, color };
    }

    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

pub fn sample_vertices() -> Vec<Vertex> {
    vec![
        Vertex {
            position: [-0.0868241, 0.49240386, 0.0],
            color: [0.5, 0.0, 0.5],
        }, // A
        Vertex {
            position: [-0.49513406, 0.06958647, 0.0],
            color: [0.5, 0.0, 0.5],
        }, // B
        Vertex {
            position: [-0.21918549, -0.44939706, 0.0],
            color: [0.5, 0.0, 0.5],
        }, // C
        Vertex {
            position: [0.35966998, -0.3473291, 0.0],
            color: [0.5, 0.0, 0.5],
        }, // D
        Vertex {
            position: [0.44147372, 0.2347359, 0.0],
            color: [0.5, 0.0, 0.5],
        },
    ]
}

pub fn sample_indices() -> Vec<u16> {
    vec![0, 1, 4, 1, 2, 4, 2, 3, 4]
}
