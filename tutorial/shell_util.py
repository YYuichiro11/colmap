import os
import subprocess
from subprocess import CalledProcessError
import shlex
    
def run_cmd_get_output(cmd, echo=False):
    """
     limitation: cannot see progress until the process is finished. Not suitable for long-running task. 
    """
    if echo:
        print("$ ", cmd)
    try:       
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        output = output.decode('utf-8')
    except CalledProcessError as e:
        print(e)
        print("output: {}".format(e.output.decode('utf-8')))
    return output
        
def run_cmd(command, echo=False):
    """
    get stdout realtime. 
    """
    if echo:
        print("$ ", command)
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if not output:
            break
        if output:
            print(output.strip().decode('utf-8'))
    rc = process.poll()
    return rc
