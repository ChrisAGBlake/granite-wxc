import boto3
from pathlib import Path
import time
import logging
import subprocess

log = logging.getLogger(__name__)

def connect_s3(profile_name):
    """
    create an s3 client
    """
    if profile_name is not None:
        sess = boto3.session.Session(profile_name=profile_name)
        s3 = sess.client('s3')
    else:
        s3 = boto3.client('s3')
    return s3

def upload_files_s3(files, s3_bucket, s3_prefix, profile_name=None, ignore_file_path=False, s3_client=None):
    """
    upload files to s3
    """
    if s3_client is None:
        s3 = connect_s3(profile_name)
    else:
        s3 = s3_client
    for i, file in enumerate(files):
        file = Path(file)
        if ignore_file_path:
            name = file.name
        else:
            name = file
        file = str(file)
        log.info(f'uploading: {file}, {i+1} of {len(files)}')
        s3.upload_file(file, s3_bucket, f'{s3_prefix}/{name}')

def upload_file_s3(file, s3_bucket, s3_name, profile_name=None, s3_client=None):
    """
    upload a file to s3
    """
    if s3_client is None:
        s3 = connect_s3(profile_name)
    else:
        s3 = s3_client
    file = str(file)
    s3.upload_file(file, s3_bucket, s3_name)

def list_files_s3(s3_bucket, s3_prefix, profile_name=None, s3_client=None):
    """
    list the files present in a given s3 bucket with the specified prefix
    """
    
    # get the list of files to download
    if s3_client is None:
        s3 = connect_s3(profile_name)
    else:
        s3 = s3_client
    keys = []
    response = s3.list_objects(Bucket=s3_bucket, Prefix=s3_prefix)
    if 'Contents' in response:
        for obj in response['Contents']:
            keys.append(obj['Key'])
        while response['IsTruncated']:
            response = s3.list_objects(Bucket=s3_bucket, Prefix=s3_prefix, Marker=keys[-1])
            for obj in response['Contents']:
                keys.append(obj['Key'])
                
    return keys

def download_files_s3(local_dir, s3_bucket, s3_prefix, keys=None, file_filter=None, profile_name=None, s3_client=None):
    """
    download files in a given folder from s3
    """

    # get the list of files to download
    if s3_client is None:
        s3 = connect_s3(profile_name)
    else:
        s3 = s3_client
    if keys is None:
        keys = []
        response = s3.list_objects(Bucket=s3_bucket, Prefix=s3_prefix)
        if 'Contents' in response:
            for obj in response['Contents']:
                keys.append(obj['Key'])
            while response['IsTruncated']:
                response = s3.list_objects(Bucket=s3_bucket, Prefix=s3_prefix, Marker=keys[-1])
                for obj in response['Contents']:
                    keys.append(obj['Key'])

    # download the trained model and associated files
    for i, key in enumerate(keys):
        log.info(f'downloading: {key}, {i+1} of {len(keys)}')
        filename = Path(key[key.rfind('/')+1:])
        if local_dir is not None:
            filename = Path(local_dir) / filename
        if file_filter is None or str(filename).find(file_filter) >= 0:
            s3.download_file(s3_bucket, key, str(filename))

def download_file_s3(s3_bucket, s3_filename, local_filename, profile_name=None, s3_client=None):
    """
    download a file from s3
    """
    if s3_client is None:
        s3 = connect_s3(profile_name)
    else:
        s3 = s3_client
    s3.download_file(s3_bucket, s3_filename, str(local_filename))

def copy_file_s3(src_bucket, src_filename, dst_bucket, dst_filename, profile_name=None, s3_client=None):
    """
    copy a file in s3
    """
    if s3_client is None:
        s3 = connect_s3(profile_name)
    else:
        s3 = s3_client
    s3.copy_object(Bucket=dst_bucket, CopySource=f'{src_bucket}/{src_filename}', Key=dst_filename)

def rename_file_s3(src_bucket, src_filename, dst_bucket, dst_filename, profile_name=None, s3_client=None):
    """
    rename / move a file in s3
    """
    if s3_client is None:
        s3 = connect_s3(profile_name)
    else:
        s3 = s3_client
    s3.copy_object(Bucket=dst_bucket, CopySource=f'{src_bucket}/{src_filename}', Key=dst_filename)
    s3.delete_object(Bucket=src_bucket, Key=src_filename)

def list_s3_directories(s3_bucket, s3_prefix, profile_name=None, s3_client=None):
    """
    list all the directories in a given bucket with the set prefix
    """
    if s3_client is None:
        s3 = connect_s3(profile_name)
    else:
        s3 = s3_client
    dirs = []
    response = s3.list_objects(Bucket=s3_bucket, Prefix=s3_prefix, Delimiter='/')
    if 'CommonPrefixes' in response:
        for obj in response['CommonPrefixes']:
            path = obj['Prefix'][:-1]
            if path.find('/') > 0:
                path = path[path.rfind('/')+1:]
            dirs.append(path)
    return dirs

def launch_instance(ami, ins_type, use_spot, key_name):
    """
    Launch an EC2 instance
    """

    # launch an instance
    log.info('launching instance')
    ec2 = boto3.client('ec2')
    if use_spot:
        res = ec2.run_instances(
            ImageId = ami,
            InstanceType = ins_type,
            InstanceMarketOptions = {'MarketType': 'spot'},
            MaxCount = 1,
            MinCount = 1,
            KeyName=key_name
        )
    else:
        res = ec2.run_instances(
            ImageId = ami,
            InstanceType = ins_type,
            MaxCount = 1,
            MinCount = 1,
            KeyName=key_name
        )
    ins_id = res['Instances'][0]['InstanceId']
    log.info(f'id of launched instance: {ins_id}')

    # wait until it's ready
    while True:
        time.sleep(10)
        ins = boto3.resource('ec2').Instance(ins_id)
        state = ins.state['Name']
        if state == 'running':
            break
        else:
            log.info(f'waiting for instance to be ready, current state: {state}')
    ip = ins.public_ip_address
    while True:
        time.sleep(10)
        res = subprocess.call(['ssh', '-o', 'StrictHostKeyChecking=no', f'ubuntu@{ip}', 'echo ready'])
        if res == 0:
            break
        else:
            log.info('waiting for ssh access to instance')
    log.info(f'ip of launched instance: {ip}')
    
    return ins_id, ip

def upload_files_instance(ip, files):
    """
    Upload files to an EC2 instance
    """
    for file in files:
        # create the directories for the file if required
        e = file.find('/')
        while e > 0:
            folder = file[:e]
            subprocess.call(['ssh', f'ubuntu@{ip}', 'mkdir', folder])
            e = file.find('/', e + 1)

        # upload the file
        subprocess.call(['scp', file, f'ubuntu@{ip}:~/{file}'])

def run_cmd(ip, cmd, in_tmux=True):
    """
    run a command within a tmux session on an EC2 instance
    """
    if in_tmux:
        subprocess.call(['ssh', f'ubuntu@{ip}', 'tmux', 'new-session', '-d', cmd])
    else:
        subprocess.call(['ssh', f'ubuntu@{ip}', cmd])

def monitor_spot_instance(ami, ins_type, ins_id, files, cmd):
    """
    monitor a spot instance,
    if it get's shutdown, wait until a new one is ready
    then launch it and continue the code that was running
    """

    while True:
        
        # check the current state of the instance
        try:
            ins = boto3.resource('ec2').Instance(ins_id)
        except:
            state = ''
        else:
            state = ins.state['Name']
        if state != 'running':
            
            # the instance is no longer running, try to launch a new instance
            try:
                new_id, ins_ip = launch_instance(ami, ins_type, True)
            except:
                log.info('unable to launch a new spot instance, trying again in 10 minutes')
            else:
                upload_files_instance(ins_ip, files)
                run_cmd(ins_ip, cmd)
                ins_id = new_id

        # wait 10 minutes
        time.sleep(600)

