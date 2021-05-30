

# This singleton class simulates an enum that consists of the GPU devices we
# support. Users can access a device using its identifier (e.g., Device.V100).
class _Device:
    def __init__(self):
        self._devices = None

    def __getattr__(self, device_name):
        if self._devices is None:
            # Lazily load the devices on the first access
            self._load_devices()
        return self._devices[device_name]

    def _load_devices(self):
        import yaml
        import habitat.habitat_cuda as hc
        import habitat.data as hd
        with open(hd.path_to_data('devices.yml')) as devices_yaml:
            devices = yaml.load(devices_yaml, Loader=yaml.Loader)
        self._devices = {
            device_name: hc.DeviceProperties(name=device_name, **properties)
            for device_name, properties in devices.items()
        }
