"""Long-lived Isaac Sim session with a command socket.

server.py and scene.py import isaaclab at module scope, so they can only be
imported after AppLauncher has started Kit. Nothing is re-exported here on
purpose — an eager import would drag isaaclab in the moment anyone touched
this package from the host.
"""
