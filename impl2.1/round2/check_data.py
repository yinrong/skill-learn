import json                                                                             
import sys
for i,line in enumerate(open(sys.argv[1])):                                                     
    s = json.loads(line)                                
    print(f'\n=== Sample {i}: violations={s["ground_truth"]["violations"]} ===')                             
    print(s['output'][:600])                            
    input('--- press Enter ---') 