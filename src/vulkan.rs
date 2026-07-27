use ash::vk;
use std::ffi::CString;
use std::ptr;

pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl VulkanContext {
    pub fn init() -> Self {
        let entry = unsafe {
            ash::Entry::load().expect("failed to load vulkan entry")
        };

        let app_name = CString::new("laguna-raw").unwrap();
        let engine_name = CString::new("laguna-raw").unwrap();

        let appinfo = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&engine_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_2);

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("instance creation failed")
        };

        let pdevices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("physical device enumeration failed")
        };

        if pdevices.is_empty() {
            panic!("no vulkan physical devices found");
        }

        let pdevice = pdevices[0];
        let queue_family_index = {
            let qprops = unsafe { instance.get_physical_device_queue_family_properties(pdevice) };
            qprops
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .expect("no compute queue found") as u32
        };

        let device_features = vk::PhysicalDeviceFeatures::default();

        let queue_priority = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priority);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_features(&device_features);

        let device = unsafe {
            instance
                .create_device(pdevice, &device_create_info, None)
                .expect("device creation failed")
        };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let memory_properties = unsafe { instance.get_physical_device_memory_properties(pdevice) };

        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe {
            device
                .create_command_pool(&command_pool_info, None)
                .expect("command pool creation failed")
        };

        Self {
            entry,
            instance,
            device,
            queue,
            queue_family_index,
            command_pool,
            memory_properties,
        }
    }

    pub fn create_shader_module(&self, spirv: &'static [u8]) -> vk::ShaderModule {
        let words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        let create_info = vk::ShaderModuleCreateInfo::default().code(&words);

        unsafe {
            self.device
                .create_shader_module(&create_info, None)
                .expect("shader module creation failed")
        }
    }

    pub fn create_descriptor_set_layout(&self, count: usize) -> vk::DescriptorSetLayout {
        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = Vec::with_capacity(count);
        for i in 0..count {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);

        unsafe {
            self.device
                .create_descriptor_set_layout(&layout_info, None)
                .expect("descriptor set layout creation failed")
        }
    }

    pub fn create_descriptor_pool(&self, count: usize) -> vk::DescriptorPool {
        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: count as u32,
        };

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(count as u32);

        unsafe {
            self.device
                .create_descriptor_pool(&pool_info, None)
                .expect("descriptor pool creation failed")
        }
    }

    pub fn allocate_descriptor_set(
        &self,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
    ) -> vk::DescriptorSet {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(std::slice::from_ref(&layout));

        unsafe {
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .expect("descriptor set allocation failed")
                .pop()
                .unwrap()
        }
    }

    pub fn write_buffer_descriptor(
        &self,
        set: vk::DescriptorSet,
        binding: usize,
        buffer: vk::Buffer,
        size: u64,
    ) {
        let buffer_info = vk::DescriptorBufferInfo {
            buffer,
            offset: 0,
            range: size,
        };

        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(binding as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info));

        unsafe {
            self.device.update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }
    }

    pub fn create_pipeline_layout(
        &self,
        set_layout: vk::DescriptorSetLayout,
        push_size: usize,
    ) -> vk::PipelineLayout {
        let set_layouts = [set_layout];
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_size as u32);

        let pipeline_layout_info = if push_size > 0 {
            vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(std::slice::from_ref(&push_range))
        } else {
            vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
        };

        unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("pipeline layout creation failed")
        }
    }

    pub fn create_compute_pipeline(
        &self,
        shader: vk::ShaderModule,
        layout: vk::PipelineLayout,
    ) -> vk::Pipeline {
        let entry_name = CString::new("main").unwrap();

        let shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(&entry_name);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(shader_stage_info)
            .layout(layout);

        unsafe {
            self.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .expect("compute pipeline creation failed")
                .pop()
                .unwrap()
        }
    }

    pub fn submit_compute(
        &self,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        set: vk::DescriptorSet,
        push: &[u8],
        dims: (u32, u32, u32),
    ) {
        let command_buffer_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe {
            self.device
                .allocate_command_buffers(&command_buffer_alloc_info)
                .expect("command buffer allocation failed")
        };

        let command_buffer = command_buffers[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("begin command buffer failed");

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                std::slice::from_ref(&set),
                &[],
            );

            if !push.is_empty() {
                self.device.cmd_push_constants(
                    command_buffer,
                    layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }

            self.device
                .cmd_dispatch(command_buffer, dims.0, dims.1, dims.2);

            self.device
                .end_command_buffer(command_buffer)
                .expect("end command buffer failed");
        }

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::default());
        let fence = unsafe {
            self.device
                .create_fence(&fence_info, None)
                .expect("fence creation failed")
        };

        let submit_info = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&command_buffer));

        unsafe {
            self.device
                .queue_submit(self.queue, std::slice::from_ref(&submit_info), fence)
                .expect("queue submit failed");

            self.device
                .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
                .expect("wait for fences failed");
        }

        unsafe {
            self.device.destroy_fence(fence, None);
        }
    }
}

pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub size: u64,
    memory: vk::DeviceMemory,
    device: ash::Device,
}

impl GpuBuffer {
    pub fn new(
        ctx: &VulkanContext,
        size: u64,
        usage: vk::BufferUsageFlags,
        mem_flags: vk::MemoryPropertyFlags,
    ) -> Self {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            ctx.device
                .create_buffer(&buffer_info, None)
                .expect("buffer creation failed")
        };

        let mem_reqs = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };

        let mem_type_index = find_memory_type(
            &ctx.memory_properties,
            mem_reqs.memory_type_bits,
            mem_flags,
        )
        .expect("failed to find suitable memory type");

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe {
            ctx.device
                .allocate_memory(&alloc_info, None)
                .expect("memory allocation failed")
        };

        unsafe {
            ctx.device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("bind buffer memory failed")
        };

        Self {
            buffer,
            size,
            memory,
            device: ctx.device.clone(),
        }
    }

    pub fn upload(&self, bytes: &[u8]) {
        unsafe {
            let data = self
                .device
                .map_memory(
                    self.memory,
                    0,
                    bytes.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("map memory failed");

            if !data.is_null() {
                ptr::copy_nonoverlapping(bytes.as_ptr(), data as *mut u8, bytes.len());
            }

            self.device.unmap_memory(self.memory);
        }
    }

    pub fn read_f32(&self, n: usize) -> Vec<f32> {
        let byte_len = (n * 4).min(self.size as usize);
        let mut result = vec![0.0f32; n];

        unsafe {
            let data = self
                .device
                .map_memory(
                    self.memory,
                    0,
                    byte_len as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("map memory failed");

            if !data.is_null() {
                let src = std::slice::from_raw_parts(data as *const u8, byte_len);
                for i in 0..n.min(byte_len / 4) {
                    let offset = i * 4;
                    result[i] = f32::from_le_bytes([
                        src[offset],
                        src[offset + 1],
                        src[offset + 2],
                        src[offset + 3],
                    ]);
                }
            }

            self.device.unmap_memory(self.memory);
        }

        result
    }
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..props.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && (props.memory_types[i as usize].property_flags & properties) == properties
        {
            return Some(i);
        }
    }
    None
}
