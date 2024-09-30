from typing import List, Callable
from hypha_rpc import login, connect_to_server
from config import Config

class Hypha:

    @staticmethod
    async def _connect(token: str):
        return await connect_to_server(
        {
            "server_url": Config.server_url,
            "workspace": Config.workspace_name,
            "client_id": Config.client_id,
            "name": "Berzelius",
            "token": token,
        }
    )

    @staticmethod
    async def _login():
        return await login({"server_url": Config.server_url})
    
    @staticmethod
    async def authenticate():
        return await Hypha._connect(await Hypha._login())

    @staticmethod
    def _get_services(callbacks: List[Callable]):
        services = {
            "name": Config.service_name,
            "id": Config.service_id,
            "config": {
                "visibility": "public",
                "require_context": True,
            }
        }
        for callback in callbacks:
            services[callback.__name__] = callback
        return services

    @staticmethod
    async def register_service(server_handle, callbacks: List[Callable]):
        return await server_handle.register_service(Hypha._get_services(callbacks), {"overwrite": True})

    @staticmethod
    def print_services(service_info, callbacks: List[Callable]):
        sid = service_info["id"]
        assert sid == f"{Config.workspace_name}/{Config.client_id}:{Config.service_id}"
        print(f"Registered service with ID: {sid}")
        for callback in callbacks:
            print(f"Test the service at: {Config.server_url}/{Config.workspace_name}/services/{Config.client_id}:{Config.service_id}/{callback.__name__}")
