import numpy as np
import pandas as pd
import re
from get_cost import calculate_die_cost,calculate_total_cost
from network_dataclass import *
from cal_perf_phy_net import *
def calculate_accelerator_areas(accelerator_list_str, csv_filename):
    """
    Calculate areas of accelerators based on the provided list and CSV file.
    
    Parameters:
    accelerator_list_str (str): A string describing a list of accelerators
    csv_filename (str): Name of the CSV file containing accelerator data
    
    Returns:
    tuple: (list of areas, sum of areas)
    """
    # Parse the accelerator list string to extract accelerator IDs
    accelerator_ids = []
    
    # Use regex to find all Function objects and extract their IDs
    pattern = r"Function\(.*?id='(.*?)'.*?\)"
    matches = re.findall(pattern, accelerator_list_str)
    
    for match in matches:
        # Clean up the ID by removing any ** characters
        clean_id = match.replace('**', '')
        accelerator_ids.append(clean_id)
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_filename, sep=',')
    except Exception as e:
        return f"Error reading CSV file: {e}", 0
    
    # Extract areas for each accelerator
    areas = []
    for acc_id in accelerator_ids:
        # Parse the accelerator ID to extract components
        # Format is typically: arch_target + glb + pe_x + pe_y + @ + tensor parallelism + @ + something + @ + something
        parts = acc_id.split('@')
        if len(parts) < 2:
            areas.append(None)
            continue
        
        # Extract the architecture part
        arch_parts = parts[0].strip().split('glb')
        if len(arch_parts) < 2:
            areas.append(None)
            print("len(arch_parts) < 2")
            continue
        
        arch_target = arch_parts[0].strip()
        
        # Extract scaling factors
        scales_part = arch_parts[1]
        scales_match = re.search(r'(\d+(?:\.\d+)?)pe_x(\d+(?:\.\d+)?)pe_y(\d+(?:\.\d+)?)', scales_part)

        if not scales_match:
            areas.append(None)
            print("if not scales_match")
            continue
        
        glb_scale = float(scales_match.group(1))
        pe_x_scale = float(scales_match.group(2))
        pe_y_scale = float(scales_match.group(3))
                
        
        # Extract tp
        try:
            tp_degree = int(parts[1].strip())
        except:
            tp_degree = 0
            print(arch_target,glb_scale,pe_x_scale,pe_y_scale)
            exit(-1)
        # Find matching row in the CSV

        matches = df[(df['arch_target'] == arch_target) & 
                     (df['glb_scale'] == glb_scale) & 
                     (df['pe_x_scale'] == pe_x_scale) & 
                     (df['pe_y_scale'] == pe_y_scale) & 
                     (df['mapper_idx'] == 0)]

        
        if not matches.empty:
            # Get the first matching area
            area = matches.iloc[0]['area']
            #for i in range(tp_degree):
            areas.append(area*tp_degree)
        else:
            areas.append(None)
            print(arch_target,glb_scale,pe_x_scale,pe_y_scale)
            exit(-1)
    
    # Calculate the sum of areas, ignoring None values
    valid_areas = [area for area in areas if area is not None]
    total_area = sum(valid_areas) if valid_areas else 0
    
    return areas, total_area

# Example usage
if __name__ == "__main__":
    # Example accelerator list string
    accelerator_list_str = """{'min_val': np.float64(0.000592394441065519), 'functions': [Function(x1=np.float64(0.0008081970918718003), a=np.float64(0.040768681978316505), b=np.float64(4.334148361596034e-05), id='simple_output_stationaryglb2pe_x0.5pe_y2@4@16@0'), Function(x1=np.float64(0.0004421722320845288), a=np.float64(0.04072718055313132), b=np.float64(1.9036467497385976e-05), id='simple_output_stationaryglb2pe_x0.5pe_y4@2@64@0'), Function(x1=np.float64(0.0008911359999999998), a=np.float64(0.011216687462222224), b=np.float64(1.4383983399814001e-05), id='simple_output_stationaryglb2pe_x0.5pe_y2@2@32@1'), Function(x1=np.float64(0.0001234832142739701), a=np.float64(0.01100168092181818), b=np.float64(4.278222249699185e-05), id='simba_likeglb4pe_x1pe_y2@2@32@1'), Function(x1=1.1664e-05, a=np.float64(0.040754847085909096), b=np.float64(1.0127733748647614e-06), id='eyeriss_likeglb4pe_x2pe_y2@1@32@1'), Function(x1=np.float64(0.000424608), a=np.float64(0.022270024669924244), b=np.float64(1.2664014134705423e-05), id='simple_output_stationaryglb1pe_x0.5pe_y4@2@16@1'), Function(x1=np.float64(0.00097344), a=np.float64(0.005703862790244107), b=np.float64(5.2960400205911635e-06), id='simple_output_stationaryglb0.5pe_x0.5pe_y4@1@16@0'), Function(x1=np.float64(1.3459650225844631e-05), a=np.float64(0.02233013477909091), b=np.float64(1.2369353949154548e-06), id='eyeriss_likeglb1pe_x4pe_y1@1@4@1'), Function(x1=0.000379456, a=np.float64(0.022270827621313133), b=np.float64(1.107920670335334e-05), id='simple_output_stationaryglb1pe_x0.5pe_y2@2@16@1'), Function(x1=0.0009031680000000001, a=np.float64(0.011189340457013891), b=np.float64(1.2173873148859029e-05), id='simple_output_stationaryglb0.5pe_x0.5pe_y1@2@16@1'), Function(x1=0.00017305600000000002, a=np.float64(0.00574794662994318), b=np.float64(3.3592912165381815e-06), id='eyeriss_likeglb2pe_x0.5pe_y2@1@16@0'), Function(x1=2.5088e-05, a=np.float64(0.022265050711590913), b=1.9963833911980804e-06, id='simple_output_stationaryglb0.5pe_x0.5pe_y4@2@16@0'), Function(x1=0.0009031680000000001, a=np.float64(0.08111186700246846), b=np.float64(1.220737479045903e-05), id='simple_output_stationaryglb0.5pe_x0.5pe_y1@2@16@1'), Function(x1=0.0008652800000000001, a=np.float64(0.00586477280181818), b=np.float64(6.860148641507782e-05), id='eyeriss_likeglb0.5pe_x0.5pe_y4@1@16@0')], 'idx': 0}"""

    net_name='squeezenet'

    network = VirtualNetwork(net_name)
    network.load_from_dir(os.path.join(NET_DIR, net_name))
    num_layers = len(network.layers)
    gene = {
        'binary_string':'10010010100110010110011111',
        'parallelism': [1]*num_layers,
        'tile_size':[1]*num_layers
    }
    physical_network = create_physical_network_from_gene(network, gene)
    
    fusion_group_mem_dict = {}
    fusion_group_mem_spec_dict = {}
    
    # Memory calculation
    num_group = len(physical_network.fusion_groups)
    for group_idx, fusion_group in enumerate(physical_network.fusion_groups):
        fusion_group_mem_dict[group_idx] = 0
        in_mem, out_mem = cal_mem_req_for_fusion_group(net_name, fusion_group)
        # first one in the group no need to account for in
        if num_group == 1:
            continue
        if group_idx == 0:
            fusion_group_mem_dict[group_idx] += out_mem
        # last one in the group no need to account for out
        elif group_idx == num_group -1:
            fusion_group_mem_dict[group_idx-1] += in_mem
        else:
            fusion_group_mem_dict[group_idx] += out_mem
            fusion_group_mem_dict[group_idx-1] += in_mem
            
    for group_idx, fusion_group in enumerate(physical_network.fusion_groups):
        if num_group == 1:
            continue
        # fusion_group_mem_spec_dict[num_group-1]=get_memory_spec(0)
        fusion_group_mem_spec_dict[group_idx] = get_memory_spec(fusion_group_mem_dict[group_idx])
        print(fusion_group_mem_spec_dict[group_idx])

    sram_area = []
    dram_cost = 0
    additional_total_area = 0
    for group_idx, fusion_group in enumerate(physical_network.fusion_groups):
        if group_idx != num_group -1:
            if fusion_group_mem_spec_dict[group_idx]['mem_type']=='sram':
                sram_area.append(f"{2*fusion_group_mem_spec_dict[group_idx]['area']},sram")
                additional_total_area+=2*fusion_group_mem_spec_dict[group_idx]['area']
            else:
                dram_cost += 2*fusion_group_mem_spec_dict[group_idx]['cost']
    # Call the function with the example data
    areas, total_area = calculate_accelerator_areas(accelerator_list_str, "final_database.csv")

    final_area_tmp = [f"{area/1e6},others" for area in areas]
    final_areas = []
    for i in range(1):
        final_areas.extend(final_area_tmp)
        final_areas.extend(sram_area)
    print(final_areas)
    chiplet_cost, yield_die_final_chiplet = calculate_total_cost(final_areas)
    chiplet_cost += dram_cost
    final_mono_area = [f"{(total_area)/1e6+additional_total_area},others"]
    mono_cost,yield_die_final_mono = calculate_total_cost(final_mono_area)
    mono_cost += dram_cost   
    print("Areas:", final_areas)
    print("Cost:", chiplet_cost)
    print("Total area:", final_mono_area[0])
    print(yield_die_final_mono,yield_die_final_chiplet)
    print(f"Cost: {mono_cost}, {mono_cost/chiplet_cost}x more expensive, {yield_die_final_chiplet/yield_die_final_mono}")
    