

def calculate_wave_info(kernel, origin_device, dest_device, metadata_manager):
    origin_occupancy = kernel.thread_block_occupancy(origin_device)
    origin_wave_size = origin_device.num_sms * origin_occupancy

    dest_registers_per_thread = metadata_manager.kernel_registers_for(
        kernel,
        dest_device,
    )
    if dest_registers_per_thread is not None:
        dest_occupancy = kernel.thread_block_occupancy(
            dest_device,
            dest_registers_per_thread,
        )
    else:
        dest_occupancy = kernel.thread_block_occupancy(dest_device)
    dest_wave_size = dest_device.num_sms * dest_occupancy

    return origin_wave_size, dest_wave_size, origin_occupancy, dest_occupancy
