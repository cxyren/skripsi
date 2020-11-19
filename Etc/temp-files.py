import tempfile
import psutil

print(tempfile.gettempdir())
print(psutil.virtual_memory())