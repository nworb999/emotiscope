import os
import requests
import json
import paramiko
from paramiko import SSHClient, Ed25519Key
import schedule
import time
from scp import SCPClient
from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder


tunnel = None
PROMPTS = []

load_dotenv()

imagination_ip = os.environ.get("IMAGINATION_IP")
imagination_port = int(os.environ.get("IMAGINATION_PORT"))
local_port = int(os.environ.get("LOCAL_PORT"))
ssh_user = os.environ.get("SSH_USERNAME")
ssh_keyfile = os.environ.get("SSH_KEYFILE")
pkey_passphrase = os.environ.get("PKEY_PASSPHRASE")
ssh_password = os.environ.get("SSH_PASSWORD")
remote_path = '/home/emma/code/emotiscope/outputs'
local_path = './landing_bay'
download_port = 22

pkey = Ed25519Key.from_private_key_file(ssh_keyfile, password=pkey_passphrase)


def fetch_new_images(remote_path, local_path, hostname, port, username, password):
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, port=port, username=username, password=password)

    scp = SCPClient(ssh.get_transport())

    stdin, stdout, stderr = ssh.exec_command(f'ls {remote_path}')
    remote_files = stdout.read().splitlines()

    existing_files = os.listdir(local_path)

    for file in remote_files:
        if file.decode('utf-8') not in existing_files:
            remote_file_path = os.path.join(remote_path, file.decode('utf-8'))
            local_file_path = os.path.join(local_path, file.decode('utf-8'))
            scp.get(remote_file_path, local_file_path)
            print(f"Copied {file.decode('utf-8')} to {local_path}")
    
    scp.close()
    ssh.close()


def main(poll_time=15):
    schedule.every(poll_time).seconds.do(fetch_new_images, remote_path, local_path, imagination_ip, download_port, ssh_user, ssh_password)
    
    while True:
        schedule.run_pending()
        time.sleep(1)
    return
    


main(1)
