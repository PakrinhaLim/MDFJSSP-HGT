import json
import os
class PCBDataGenerator:
    def __init__(self, number_pieces_per_product = 1000, batch_size=100):
        # Define templates for the 5 Multilayer PCB Types
        # Times are represented in abstract "time units" (e.g., minutes).
        # Note: For Batch Processing Machines (like Lamination Press), the time is fixed regardless of batch size.
        self.number_pieces_per_product = number_pieces_per_product
        self.batch_size = batch_size
        self.num_batches = number_pieces_per_product // batch_size
        self._multilyer_product_templates = [
        {
            "type": "Product_1_Standard_4Layer",
            "operations": [
                # Op0: DES Line (M0-M3) - 0.2 mins/pc
                {"processing_times": {f"machine_{i}": int(0.2 * self.batch_size) for i in range(0, 4)}, "predecessors": None},
                # Op1: AOI (M4-M7) - 0.1 mins/pc
                {"processing_times": {f"machine_{i}": int(0.1 * self.batch_size) for i in range(4, 8)}, "predecessors": [0]},
                # Op2: Lamination Press (M8-M10) - Batch machine, always takes 180 mins regardless of batch
                {"processing_times": {f"machine_{i}": 180 for i in range(8, 11)}, "predecessors": [1]},
                # Op3: CNC Drilling (M11-M26) - 0.5 mins/pc
                {"processing_times": {f"machine_{i}": int(0.5 * self.batch_size) for i in range(11, 27)}, "predecessors": [2]},
                # Op4: Plating (M27-M28) - 0.3 mins/pc
                {"processing_times": {f"machine_{i}": int(0.3 * self.batch_size) for i in range(27, 29)}, "predecessors": [3]},
                # Op5: Solder Mask (M29-M32) - 0.4 mins/pc + 30 mins fixed baking
                {"processing_times": {f"machine_{i}": int(0.4 * self.batch_size) + 30 for i in range(29, 33)}, "predecessors": [4]},
                # Op6: CNC Router (M33-M36) - 0.2 mins/pc
                {"processing_times": {f"machine_{i}": int(0.2 * self.batch_size) for i in range(33, 37)}, "predecessors": [5]},
                # Op7: Electrical Testing (M37-M39) - 0.3 mins/pc
                {"processing_times": {f"machine_{i}": int(0.3 * self.batch_size) for i in range(37, 40)}, "predecessors": [6]},
            ]
        },
        {
            "type": "Product_2_Complex_8Layer",
            # Needs double pressing and much longer drilling times for HDI (High Density Interconnect)
            "operations": [
                {"processing_times": {f"machine_{i}": int(0.3 * self.batch_size) for i in range(0, 4)}, "predecessors": None}, # DES
                {"processing_times": {f"machine_{i}": int(0.15 * self.batch_size) for i in range(4, 8)}, "predecessors": [0]}, # AOI
                {"processing_times": {f"machine_{i}": 240 for i in range(8, 11)}, "predecessors": [1]}, # First Press
                {"processing_times": {f"machine_{i}": int(1.2 * self.batch_size) for i in range(11, 27)}, "predecessors": [2]}, # Drill 1
                {"processing_times": {f"machine_{i}": int(0.4 * self.batch_size) for i in range(27, 29)}, "predecessors": [3]}, # Plating
                {"processing_times": {f"machine_{i}": 180 for i in range(8, 11)}, "predecessors": [4]}, # Second Press (HDI)
                {"processing_times": {f"machine_{i}": int(1.5 * self.batch_size) for i in range(11, 27)}, "predecessors": [5]}, # Drill 2
                {"processing_times": {f"machine_{i}": int(0.5 * self.batch_size) for i in range(27, 29)}, "predecessors": [6]}, # Plating 2
                {"processing_times": {f"machine_{i}": int(0.6 * self.batch_size) + 30 for i in range(29, 33)}, "predecessors": [7]}, # Solder Mask
                {"processing_times": {f"machine_{i}": int(0.3 * self.batch_size) for i in range(33, 37)}, "predecessors": [8]}, # Router
                {"processing_times": {f"machine_{i}": int(0.5 * self.batch_size) for i in range(37, 40)}, "predecessors": [9]}, # Testing
            ]
        },
        {
            "type": "Product_3_Fast_Turnaround_2Layer",
            # Skips lamination press completely
            "operations": [
                {"processing_times": {f"machine_{i}": int(0.2 * self.batch_size) for i in range(11, 27)}, "predecessors": None}, # Drill right away
                {"processing_times": {f"machine_{i}": int(0.25 * self.batch_size) for i in range(27, 29)}, "predecessors": [0]}, # Plating
                {"processing_times": {f"machine_{i}": int(0.15 * self.batch_size) for i in range(0, 4)}, "predecessors": [1]}, # DES outer layers
                {"processing_times": {f"machine_{i}": int(0.1 * self.batch_size) for i in range(4, 8)}, "predecessors": [2]}, # AOI
                {"processing_times": {f"machine_{i}": int(0.4 * self.batch_size) + 20 for i in range(29, 33)}, "predecessors": [3]}, # Solder Mask
                {"processing_times": {f"machine_{i}": int(0.15 * self.batch_size) for i in range(33, 37)}, "predecessors": [4]}, # Router
                {"processing_times": {f"machine_{i}": int(0.2 * self.batch_size) for i in range(37, 40)}, "predecessors": [5]}, # Testing
            ]
        },
        {
            "type": "Product_4_High_Density_Interconnect",
            # Very Drill-heavy board
            "operations": [
                {"processing_times": {f"machine_{i}": int(0.2 * self.batch_size) for i in range(0, 4)}, "predecessors": None}, # DES
                {"processing_times": {f"machine_{i}": int(0.2 * self.batch_size) for i in range(4, 8)}, "predecessors": [0]}, # AOI
                {"processing_times": {f"machine_{i}": 200 for i in range(8, 11)}, "predecessors": [1]}, # Press
                {"processing_times": {f"machine_{i}": int(2.5 * self.batch_size) for i in range(11, 27)}, "predecessors": [2]}, # VERY LONG DRILL
                {"processing_times": {f"machine_{i}": int(0.4 * self.batch_size) for i in range(27, 29)}, "predecessors": [3]}, # Plating
                {"processing_times": {f"machine_{i}": int(0.5 * self.batch_size) + 40 for i in range(29, 33)}, "predecessors": [4]}, # Solder Mask
                {"processing_times": {f"machine_{i}": int(0.25 * self.batch_size) for i in range(33, 37)}, "predecessors": [5]}, # Router
                {"processing_times": {f"machine_{i}": int(0.4 * self.batch_size) for i in range(37, 40)}, "predecessors": [6]}, # Testing
            ]
        },
        {
            "type": "Product_5_RF_Microwave_Board",
            # Needs extensive testing
            "operations": [
                {"processing_times": {f"machine_{i}": int(0.25 * self.batch_size) for i in range(0, 4)}, "predecessors": None}, # DES
                {"processing_times": {f"machine_{i}": int(0.15 * self.batch_size) for i in range(4, 8)}, "predecessors": [0]}, # AOI
                {"processing_times": {f"machine_{i}": 150 for i in range(8, 11)}, "predecessors": [1]}, # Press
                {"processing_times": {f"machine_{i}": int(0.6 * self.batch_size) for i in range(11, 27)}, "predecessors": [2]}, # Drill
                {"processing_times": {f"machine_{i}": int(0.35 * self.batch_size) for i in range(27, 29)}, "predecessors": [3]}, # Plating
                {"processing_times": {f"machine_{i}": int(0.4 * self.batch_size) + 30 for i in range(29, 33)}, "predecessors": [4]}, # Solder Mask
                {"processing_times": {f"machine_{i}": int(0.2 * self.batch_size) for i in range(33, 37)}, "predecessors": [5]}, # Router
                {"processing_times": {f"machine_{i}": int(1.5 * self.batch_size) for i in range(37, 40)}, "predecessors": [6]}, # HIGH TESTING
            ]
        }
    ]
        self._processing_info = None

    def _generate_pcb_data(self):
        processing_info = {
            "instance_name": f"multilayer_pcb_{self.num_batches * len(self._multilyer_product_templates)}_batches",
            "nr_machines": 40,
            "jobs": []
        }
        global_job_id = 0
        global_op_id = 0

        for prod_idx, template in enumerate(self._multilyer_product_templates):
            for batch_idx in range(self.num_batches):
                job_ops = []
                op_id_mapping = {} # local index to global index
            
            for local_op_idx, op_template in enumerate(template["operations"]):
                current_op_id = global_op_id
                op_id_mapping[local_op_idx] = current_op_id
                
                # Link local predecessors to their new globally unique IDs
                global_predecessors = None
                if op_template["predecessors"] is not None:
                    global_predecessors = [op_id_mapping[p] for p in op_template["predecessors"]]
                    
                job_ops.append({
                    "operation_id": current_op_id,
                    "processing_times": op_template["processing_times"],
                    "predecessors": global_predecessors
                })
                global_op_id += 1
                
            processing_info["jobs"].append({
                "job_id": global_job_id,
                # Extra meta data fields not strictly needed but helpful for context
                "product_type": template["type"], 
                "pieces_in_batch": self.batch_size,
                "operations": job_ops
            })
            global_job_id += 1
        return processing_info

    @property
    def processing_info(self):
        if self._processing_info is None:
            self._processing_info = self._generate_pcb_data()
        return self._processing_info
    
    def save_dataset(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self._processing_info, f, indent=4)
        print(f"Data is saved to {filepath}")


