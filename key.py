from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
# 1. 生成私钥
private_key = ec.generate_private_key(ec.SECP384R1())

# 2. 从私钥生成公钥
public_key = private_key.public_key()

   # 3. 将私钥保存到文件（！！！这个文件必须绝对保密！！！）
with open("private_key.pem", "wb") as f:
    f.write(private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
        ))
    
    # 4. 将公钥保存到文件（这个文件将要内置到你的应用里）
with open("public_key.pem", "wb") as f:
    f.write(public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
    
print("密钥对 private_key.pem 和 public_key.pem 已生成。")