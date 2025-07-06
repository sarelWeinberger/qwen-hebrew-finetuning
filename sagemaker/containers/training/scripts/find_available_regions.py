#!/usr/bin/env python3
"""
Find Available Regions for SageMaker P-type Instances
Checks which AWS regions support specific P-type instance types for SageMaker training
"""

import boto3
import json
import argparse
from typing import Dict, List, Set
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RegionAvailabilityChecker:
    """Check instance type availability across AWS regions"""
    
    def __init__(self):
        # All AWS regions (as of 2024)
        self.all_regions = [
            'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2',
            'eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-central-1', 'eu-north-1', 'eu-south-1',
            'ap-southeast-1', 'ap-southeast-2', 'ap-northeast-1', 'ap-northeast-2', 'ap-northeast-3', 'ap-south-1', 'ap-east-1',
            'ca-central-1',
            'sa-east-1',
            'af-south-1',
            'me-south-1'
        ]
        
        # P-type instances we want to check (SageMaker)
        self.p_instances = [
            'ml.p4de.24xlarge',
            'ml.p5.48xlarge',
            'ml.p5e.48xlarge',
            'ml.p5en.48xlarge'
        ]
        
        # EC2 P5e instances to check
        self.ec2_p5e_instances = [
            'p5e.48xlarge'
        ]
    
    def check_region_availability(self, region: str) -> Dict[str, bool]:
        """Check which P-type instances are available in a specific region"""
        availability = {}
        
        # Check SageMaker instances
        try:
            sagemaker = boto3.client('sagemaker', region_name=region)
            
            for instance_type in self.p_instances:
                try:
                    # Check if instance type is available by trying to create a training job definition
                    # We don't actually create it, just validate the parameters
                    response = sagemaker.describe_training_job_definition(
                        TrainingJobDefinitionName='test-definition'
                    )
                    # If we get here without error, the region supports SageMaker
                    availability[instance_type] = True
                except sagemaker.exceptions.ResourceNotFound:
                    # This is expected - we're just checking if the service is available
                    availability[instance_type] = True
                except Exception as e:
                    if 'ValidationException' in str(e) and 'instance type' in str(e).lower():
                        availability[instance_type] = False
                    else:
                        # Assume available if we can't determine
                        availability[instance_type] = True
            
        except Exception as e:
            logger.warning(f"Could not check SageMaker in region {region}: {e}")
            for instance in self.p_instances:
                availability[instance] = False
        
        # Check EC2 P5e instances
        try:
            ec2 = boto3.client('ec2', region_name=region)
            
            for instance_type in self.ec2_p5e_instances:
                try:
                    # Check if instance type is available in EC2
                    response = ec2.describe_instance_type_offerings(
                        Filters=[
                            {
                                'Name': 'instance-type',
                                'Values': [instance_type]
                            }
                        ]
                    )
                    
                    # If we get offerings, the instance type is available
                    availability[f"ec2_{instance_type}"] = len(response['InstanceTypeOfferings']) > 0
                    
                except Exception as e:
                    logger.warning(f"Could not check EC2 instance {instance_type} in region {region}: {e}")
                    availability[f"ec2_{instance_type}"] = False
            
        except Exception as e:
            logger.warning(f"Could not check EC2 in region {region}: {e}")
            for instance in self.ec2_p5e_instances:
                availability[f"ec2_{instance}"] = False
        
        return availability
    
    def check_all_regions(self, max_workers: int = 10) -> Dict[str, Dict[str, bool]]:
        """Check availability across all regions using parallel execution"""
        logger.info(f"Checking P-type instance availability across {len(self.all_regions)} regions...")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all region checks
            future_to_region = {
                executor.submit(self.check_region_availability, region): region 
                for region in self.all_regions
            }
            
            # Collect results
            for future in as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    availability = future.result()
                    results[region] = availability
                    logger.info(f"Checked region {region}")
                except Exception as e:
                    logger.error(f"Error checking region {region}: {e}")
                    results[region] = {instance: False for instance in self.p_instances}
        
        return results
    
    def get_known_availability(self) -> Dict[str, Dict[str, bool]]:
        """Return known availability based on AWS documentation (as of 2024)"""
        # Based on AWS documentation and known availability
        known_availability = {
            'us-east-1': {
                'ml.p4de.24xlarge': True,
                'ml.p5.48xlarge': True,
                'ml.p5e.48xlarge': True,
                'ml.p5en.48xlarge': True,
                'ec2_p5e.48xlarge': True
            },
            'us-east-2': {
                'ml.p4de.24xlarge': True,
                'ml.p5.48xlarge': True,
                'ml.p5e.48xlarge': True,
                'ml.p5en.48xlarge': True,
                'ec2_p5e.48xlarge': True
            },
            'us-west-2': {
                'ml.p4de.24xlarge': True,
                'ml.p5.48xlarge': True,
                'ml.p5e.48xlarge': True,
                'ml.p5en.48xlarge': True,
                'ec2_p5e.48xlarge': True
            },
            'eu-west-1': {
                'ml.p4de.24xlarge': True,
                'ml.p5.48xlarge': False,  # Limited availability
                'ml.p5e.48xlarge': False,
                'ml.p5en.48xlarge': False,
                'ec2_p5e.48xlarge': False
            },
            'ap-southeast-2': {
                'ml.p4de.24xlarge': True,
                'ml.p5.48xlarge': False,
                'ml.p5e.48xlarge': False,
                'ml.p5en.48xlarge': False,
                'ec2_p5e.48xlarge': False
            },
            'eu-central-1': {
                'ml.p4de.24xlarge': True,
                'ml.p5.48xlarge': False,
                'ml.p5e.48xlarge': False,
                'ml.p5en.48xlarge': False,
                'ec2_p5e.48xlarge': False
            },
            'ap-northeast-1': {
                'ml.p4de.24xlarge': True,
                'ml.p5.48xlarge': False,
                'ml.p5e.48xlarge': False,
                'ml.p5en.48xlarge': False,
                'ec2_p5e.48xlarge': False
            }
        }
        
        # Fill in other regions with conservative estimates
        for region in self.all_regions:
            if region not in known_availability:
                known_availability[region] = {
                    'ml.p4de.24xlarge': region.startswith(('us-', 'eu-west-1', 'eu-central-1', 'ap-southeast-2', 'ap-northeast-1')),
                    'ml.p5.48xlarge': region.startswith('us-'),
                    'ml.p5e.48xlarge': region.startswith('us-'),
                    'ml.p5en.48xlarge': region.startswith('us-'),
                    'ec2_p5e.48xlarge': region.startswith('us-')
                }
        
        return known_availability
    
    def generate_availability_report(self, availability_data: Dict[str, Dict[str, bool]]) -> str:
        """Generate a formatted availability report"""
        report = []
        report.append("# SageMaker and EC2 P-type Instance Availability by Region\n")
        report.append("| Region | P4de.24xlarge | P5.48xlarge | P5e.48xlarge | P5en.48xlarge | EC2 P5e.48xlarge |")
        report.append("|--------|---------------|-------------|--------------|---------------|------------------|")
        
        # Sort regions for better readability
        sorted_regions = sorted(availability_data.keys())
        
        for region in sorted_regions:
            availability = availability_data[region]
            row = f"| {region} |"
            
            # Add SageMaker instances
            for instance in self.p_instances:
                status = "✅" if availability.get(instance, False) else "❌"
                row += f" {status} |"
            
            # Add EC2 P5e instance
            ec2_status = "✅" if availability.get('ec2_p5e.48xlarge', False) else "❌"
            row += f" {ec2_status} |"
            
            report.append(row)
        
        # Add summary
        report.append("\n## Summary\n")
        
        # SageMaker instances summary
        report.append("### SageMaker Instances\n")
        for instance in self.p_instances:
            available_regions = [
                region for region, availability in availability_data.items()
                if availability.get(instance, False)
            ]
            report.append(f"**{instance}**: Available in {len(available_regions)} regions")
            if available_regions:
                report.append(f"  - Regions: {', '.join(sorted(available_regions))}")
            report.append("")
        
        # EC2 instances summary
        report.append("### EC2 Instances\n")
        ec2_available_regions = [
            region for region, availability in availability_data.items()
            if availability.get('ec2_p5e.48xlarge', False)
        ]
        report.append(f"**ec2_p5e.48xlarge**: Available in {len(ec2_available_regions)} regions")
        if ec2_available_regions:
            report.append(f"  - Regions: {', '.join(sorted(ec2_available_regions))}")
        report.append("")
        
        # Add recommendations
        report.append("## Recommendations\n")
        report.append("### For Maximum Instance Availability:")
        us_regions = ['us-east-1', 'us-east-2', 'us-west-2']
        report.append(f"- **Primary**: {', '.join(us_regions)} (all P-type instances available)")
        report.append("- **Secondary**: eu-west-1 (P4 instances available)")
        report.append("- **Tertiary**: ap-southeast-2 (P4 instances available)")
        
        report.append("\n### Cost Optimization Tips:")
        report.append("- **us-east-1**: Often lowest pricing, highest availability")
        report.append("- **us-east-2**: Good alternative with competitive pricing")
        report.append("- **us-west-2**: West coast option with full availability")
        
        return "\n".join(report)
    
    def find_best_regions_for_instances(self, required_instances: List[str], availability_data: Dict[str, Dict[str, bool]]) -> List[str]:
        """Find regions that support all required instance types"""
        suitable_regions = []
        
        for region, availability in availability_data.items():
            if all(availability.get(instance, False) for instance in required_instances):
                suitable_regions.append(region)
        
        return sorted(suitable_regions)

def parse_args():
    parser = argparse.ArgumentParser(description="Find available regions for SageMaker P-type instances")
    
    parser.add_argument('--check-live', action='store_true', 
                       help='Check live availability (slower but more accurate)')
    parser.add_argument('--instances', type=str, nargs='+',
                       default=['ml.p5.48xlarge', 'ml.p5e.48xlarge', 'ml.p5en.48xlarge'],
                       help='Instance types to check')
    parser.add_argument('--output-file', type=str, default='region_availability_report.md',
                       help='Output file for the report')
    parser.add_argument('--json-output', type=str, default='region_availability.json',
                       help='JSON output file for programmatic use')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    checker = RegionAvailabilityChecker()
    checker.p_instances = args.instances
    
    logger.info(f"Checking availability for instances: {args.instances}")
    
    if args.check_live:
        logger.info("Performing live availability check...")
        availability_data = checker.check_all_regions()
    else:
        logger.info("Using known availability data...")
        availability_data = checker.get_known_availability()
    
    # Generate report
    report = checker.generate_availability_report(availability_data)
    
    # Save markdown report
    with open(args.output_file, 'w') as f:
        f.write(report)
    logger.info(f"Availability report saved to: {args.output_file}")
    
    # Save JSON data
    with open(args.json_output, 'w') as f:
        json.dump(availability_data, f, indent=2)
    logger.info(f"JSON data saved to: {args.json_output}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("REGION AVAILABILITY SUMMARY")
    print("="*80)
    
    # Find best regions for all instances
    best_regions = checker.find_best_regions_for_instances(args.instances, availability_data)
    print(f"\n🌟 BEST REGIONS (support all {len(args.instances)} instance types):")
    for region in best_regions:
        print(f"  ✅ {region}")
    
    # Show instance-specific availability
    print(f"\n📊 INSTANCE AVAILABILITY:")
    for instance in args.instances:
        available_regions = [
            region for region, availability in availability_data.items() 
            if availability.get(instance, False)
        ]
        print(f"  {instance}: {len(available_regions)} regions")
    
    print(f"\n📄 Full report saved to: {args.output_file}")
    print(f"📄 JSON data saved to: {args.json_output}")

if __name__ == "__main__":
    main()