# this script will help with selecting the right gpu

import subprocess, time, sys, math

p = subprocess.Popen(["lsload -gpuload"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
stdout, stderr = p.communicate()
lines = stdout.split('\n')

hosts = {}
for line in lines[1:-1]:
    items = line.split('  ')
    items = [item.replace(' ', '') for item in items if item != '']
    if items[0].startswith('hi'):
        host = items[0]
        hosts[host] = dict(free=0, occupied=0)
        items.pop(0)
    if items[4]  == "0%" and items[5] == "0%":
        hosts[host]['free'] += 1
    else:
        hosts[host]['occupied'] += 1
classes = {"pascal_12g": ["hi-026l", "hi-027l", "hi-028l", "hi-029l"],
           "rtx_12g": ["hi-031l", "hi-032l", "hi-033l"],
           "rtx_24g": ["hi-034l"],
           "rtx_48g": ["hi-030l", "hi-035l", "hi-036l"]}
available = {"pascal_12g": 0,
             "rtx_12g": 0,
             "rtx_24g": 0,
             "rtx_48g": 0}
for host in hosts:
    for class_ in classes:
        if host in classes[class_]:
            available[class_] += hosts[host]['free']
            break
for class_ in available:
    print(available[class_], "of class", class_, "available")
