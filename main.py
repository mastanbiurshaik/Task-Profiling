import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import warnings
import json
from scipy.stats import weibull_min

class TaskProfilerEngine:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.z_threshold = 5.0  
        self.stratification_rules = {
        'short_task': {'max_duration': 500},  
        'long_task': {'min_duration': 2000},  
        'cpu_bound': {
            'cpu_threshold': 70,  
            'memory_threshold': 30
        },
        'io_bound': {
            'cpu_threshold': 30,
            'memory_threshold': 70  
        }
    }
        self.tabu_list = deque(maxlen=10)  
        self.falcon_params = {
            'high_priority_threshold': 0.8,
            'low_priority_threshold': 0.3
        }
    
    def load_data(self, file_path):
        """Load task data from CSV file"""
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} tasks from {file_path}")
        print("Original data summary:")
        print(df.describe())
        return df
    
    def normalize_features(self, df):
        """Normalize task attributes to a uniform range [0,1]"""
        if len(df) == 0:
            return df
            
        features_to_normalize = [
            'CPU_Utilization (%)',
            'Memory_Consumption (MB)',
            'Task_Execution_Time (ms)',
            'System_Throughput (tasks/sec)',
            'Task_Waiting_Time (ms)',
            'Network_Bandwidth_Utilization (Mbps)',
            'Error_Rate (%)',
            'Projection_Score',
            'Predicted_Execution_Duration (ms)',
            'Predicted_Context_Switch_Penalty (ms)',
            'Host_Cluster_Suitability_Score',
            'Scheduling_Priority_Score',
            'Envy_Freeness_Score',
            'VM_Placement_Diversity_Score',
            'PM_Utilization_Efficiency (%)',
            'Observed_Delay_Deviation (ms)',
            'Context_Switch_Overhead (%)',
            'Thermal_Impact_Score',
            'Energy_Efficiency_Index',
            'CPU_Share (%)',
            'Memory_Share (%)',
            'Bandwidth_Share (%)',
            'Envy_Rate (%)',
            'Dominant_Resource_Share (%)',
            'Memory_Utilization (%)'
        ]
        
        features_to_normalize = [col for col in features_to_normalize if col in df.columns]
        
        if not features_to_normalize:
            print("Warning: No features available for normalization")
            return df
            
        # Normalize selected features
        if len(df) > 1:
            normalized_features = self.scaler.fit_transform(df[features_to_normalize])
        else:
            if hasattr(self.scaler, 'scale_'):
                normalized_features = (df[features_to_normalize] - self.scaler.min_) / self.scaler.scale_
            else:
                print("Warning: No pre-fitted scaler available for single task normalization")
                return df
            
        df[features_to_normalize] = normalized_features
        
        df.rename(columns={col: f"Normalized_{col}" for col in features_to_normalize}, inplace=True)
        
        print("\nNormalization completed")
        print("Normalized data summary:")
        normalized_cols = [f"Normalized_{col}" for col in features_to_normalize]
        print(df[normalized_cols].describe())
        return df
    
    def filter_noise(self, df):
        if len(df) <= 1:
            print("Warning: Not enough data for noise filtering (minimum 2 samples required)")
            return df
            
        core_metrics = [
            'Normalized_CPU_Utilization (%)',
            'Normalized_Memory_Consumption (MB)',
            'Normalized_Task_Execution_Time (ms)',
            'Normalized_Network_Bandwidth_Utilization (Mbps)'
        ]
        core_metrics = [col for col in core_metrics if col in df.columns]
        
        if not core_metrics:
            print("Warning: No core metrics available for noise filtering")
            return df
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = df[core_metrics].median()
            mad = stats.median_abs_deviation(df[core_metrics], nan_policy='omit')
            modified_z_scores = 0.6745 * np.abs((df[core_metrics] - median) / mad)
        
        filtered_df = df[(modified_z_scores < self.z_threshold).all(axis=1)]
        
        print(f"\nFiltered {len(df) - len(filtered_df)} outlier tasks")
        print(f"Remaining tasks: {len(filtered_df)}")
        return filtered_df
    
    def kaczmarz_sampling(self, df, n_components=5):
        """Dimensionality reduction using deterministic Kaczmarz method"""
        if len(df) == 0:
            print("Warning: Empty dataframe received for Kaczmarz sampling")
            return df
            
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) == 0:
            print("Warning: No numerical columns found for Kaczmarz sampling")
            return df
            
        X = df[numerical_cols].values
        
        n_features = X.shape[1]
        fib = lambda n: int((0.5 + 0.5 * np.sqrt(5))**n / np.sqrt(5) + 0.5)
        projection_matrix = np.array([
            [1/(fib(i+j+2)) for j in range(n_components)]  
            for i in range(n_features)
        ])
        
        projected_features = np.dot(X, projection_matrix)
        
        for i in range(n_components):
            df[f'Projected_Feature_{i+1}'] = projected_features[:, i]
        
        print("\nApplied Kaczmarz sampling:")
        print(pd.DataFrame(projected_features).describe())
        return df
    
    def stratify_tasks(self, df):
        if 'Normalized_CPU_Utilization (%)' in df.columns:
            cpu_util = df['Normalized_CPU_Utilization (%)'] * (89.98 - 10) + 10  
            
            resource_conditions = [
                (cpu_util > 70) & (df['Normalized_Memory_Consumption (MB)'] < 0.3),  
                (cpu_util < 30) & (df['Normalized_Memory_Consumption (MB)'] > 0.7),  
                (cpu_util > 0.6) & (df['Normalized_Memory_Consumption (MB)'] > 0.6)  
            ]
            resource_labels = ['cpu_bound', 'io_bound', 'memory_intensive']
            df['Resource_Type'] = np.select(resource_conditions, resource_labels, default='balanced')
        
        return df
    
    def extract_features(self, df):
        """Feature Extraction Layer (Layer 2) - Deterministic implementation"""
        print("\nStarting Feature Extraction (Layer 2)")
        
        if len(df) == 0:
            print("Warning: Empty dataframe received for feature extraction")
            return df
            
        # WeiDiD - Weibull Duration and Dependency Analyzer
        df = self._extract_temporal_features(df)
        df = self._extract_structural_features(df)
        df = self._extract_contextual_features(df)
        df = self._extract_dag_features(df)  
        
        print("\nFeature extraction completed")
        print("Extracted features summary:")
        print(df.filter(regex='WeiDiD_').describe())
        return df
    
    def _extract_temporal_features(self, df):
        if len(df) == 0:
            return df
            
        if 'Normalized_Task_Execution_Time (ms)' not in df.columns:
            print("Warning: Missing execution time column for temporal features")
            df['WeiDiD_Execution_Shape'] = np.nan
            df['WeiDiD_Execution_Scale'] = np.nan
            df['WeiDiD_Delay_Likelihood'] = np.nan
            return df
            
        try:
            bins = pd.cut(df['Normalized_CPU_Utilization (%)'], bins=5)
            for name, group in df.groupby(bins):
                group_times = group['Normalized_Task_Execution_Time (ms)'] * self.scaler.scale_[2] + self.scaler.min_[2]
                if len(group_times) > 10:  
                    shape, loc, scale = weibull_min.fit(group_times, floc=0)
                    df.loc[group.index, 'WeiDiD_Execution_Shape'] = shape
                    df.loc[group.index, 'WeiDiD_Execution_Scale'] = scale
        except Exception as e:
            print(f"Weibull fitting failed: {str(e)}")
            df['WeiDiD_Execution_Shape'] = 1.0  
            df['WeiDiD_Execution_Scale'] = df['Normalized_Task_Execution_Time (ms)'].mean()
            
        df['WeiDiD_Delay_Likelihood'] = self._calculate_delay_likelihood(df)
        return df
        
    def _calculate_delay_likelihood(self, df):
        if len(df) == 0:
            return np.array([])
            
        delay_likelihood = np.zeros(len(df))
        
        has_cpu = 'Normalized_CPU_Utilization (%)' in df.columns
        has_mem = 'Normalized_Memory_Consumption (MB)' in df.columns
        has_net = 'Normalized_Network_Bandwidth_Utilization (Mbps)' in df.columns
        has_cs = 'Normalized_Predicted_Context_Switch_Penalty (ms)' in df.columns
        
        if has_cpu and has_mem and has_net:
            cpu_util = df['Normalized_CPU_Utilization (%)']
            mem_util = df['Normalized_Memory_Consumption (MB)']
            net_util = df['Normalized_Network_Bandwidth_Utilization (Mbps)']
            
            if has_cs:
                cs_penalty = df['Normalized_Predicted_Context_Switch_Penalty (ms)']
                delay_likelihood = 0.4*cpu_util + 0.3*mem_util + 0.2*net_util + 0.1*cs_penalty
            else:
                delay_likelihood = 0.5*cpu_util + 0.3*mem_util + 0.2*net_util
        else:
            print("Warning: Missing required columns for delay likelihood calculation")
            
        return delay_likelihood
    
    def _extract_structural_features(self, df):
        if len(df) == 0:
            return df
            
        if 'Normalized_Task_Execution_Time (ms)' in df.columns and 'Normalized_CPU_Utilization (%)' in df.columns:
            df['WeiDiD_DAG_Depth'] = (
                df['Normalized_Task_Execution_Time (ms)'] * 
                df['Normalized_CPU_Utilization (%)'] * 
                self.scaler.scale_[2] * self.scaler.scale_[0]
            )
        else:
            df['WeiDiD_DAG_Depth'] = np.nan
            
        if 'Normalized_Scheduling_Priority_Score' in df.columns:
            df['WeiDiD_Node_Criticality'] = np.sqrt(
                df['Normalized_CPU_Utilization (%)']**2 +
                df['Normalized_Memory_Consumption (MB)']**2 +
                df['Normalized_Scheduling_Priority_Score']**2
            )
        elif 'Normalized_CPU_Utilization (%)' in df.columns and 'Normalized_Memory_Consumption (MB)' in df.columns:
            df['WeiDiD_Node_Criticality'] = np.sqrt(
                df['Normalized_CPU_Utilization (%)']**2 +
                df['Normalized_Memory_Consumption (MB)']**2
            )
        else:
            df['WeiDiD_Node_Criticality'] = np.nan
            
        return df
    
    def _extract_dag_features(self, df):
        """Extract DAG topology features"""
        if len(df) == 0:
            return df
            
        if 'Dependencies' not in df.columns:
            if 'Normalized_CPU_Utilization (%)' in df.columns:
                df['WeiDiD_Fan_In'] = (df['Normalized_CPU_Utilization (%)'] * 4).astype(int) + 1
            else:
                df['WeiDiD_Fan_In'] = 1
                
            if 'Normalized_Memory_Consumption (MB)' in df.columns:
                df['WeiDiD_Fan_Out'] = (df['Normalized_Memory_Consumption (MB)'] * 4).astype(int) + 1
            else:
                df['WeiDiD_Fan_Out'] = 1
                
            if 'Normalized_Task_Waiting_Time (ms)' in df.columns:
                df['WeiDiD_Slack_Time'] = df['Normalized_Task_Waiting_Time (ms)'] * 0.5
            else:
                df['WeiDiD_Slack_Time'] = 0
                
            if 'WeiDiD_Node_Criticality' in df.columns:
                df['WeiDiD_Critical_Path'] = (df['WeiDiD_Node_Criticality'] > 0.7).astype(int)
            else:
                df['WeiDiD_Critical_Path'] = 0
        else:
            df['WeiDiD_Fan_In'] = df['Dependencies'].apply(lambda x: len(x.split(',')) if pd.notna(x) else 0)
            df['WeiDiD_Fan_Out'] = df['Dependents'].apply(lambda x: len(x.split(',')) if 'Dependents' in df.columns and pd.notna(x) else 0)
            df['WeiDiD_Slack_Time'] = (df['Deadline'] - df['Execution_Time']) if 'Deadline' in df.columns else 0
            df['WeiDiD_Critical_Path'] = (df['Critical'] == 'Yes').astype(int) if 'Critical' in df.columns else 0
        
        return df
    
    def _extract_contextual_features(self, df):
        """Extract contextual features about host affinity"""
        if len(df) == 0:
            return df
            
        if 'Normalized_Host_Cluster_Suitability_Score' in df.columns:
            df['WeiDiD_Host_Affinity'] = df['Normalized_Host_Cluster_Suitability_Score']
        elif 'Normalized_Network_Bandwidth_Utilization (Mbps)' in df.columns:
            df['WeiDiD_Host_Affinity'] = (
                1.0 - df['Normalized_Network_Bandwidth_Utilization (Mbps)']
            )
        else:
            df['WeiDiD_Host_Affinity'] = 0.5  
            
        df['WeiDiD_Load_Correlation'] = self._predict_future_load(df)
        
        return df
    
    def _predict_future_load(self, df):
        """Predict future host load using deterministic time patterns"""
        if len(df) == 0:
            return np.array([])
            
        if 'Task_Start_Time' in df.columns:
            try:
                df['Start_Datetime'] = pd.to_datetime(df['Task_Start_Time'])
                df['Hour'] = df['Start_Datetime'].dt.hour
                
                hour_pattern = np.sin(df['Hour'] * np.pi / 12) 
                return (hour_pattern + 1) / 2  
            except:
                return np.zeros(len(df))
        return np.zeros(len(df))
    
    def preprocess(self, input_file, output_file=None):
        # Step 1: Load data
        df = self.load_data(input_file)
        
        # Step 2: Normalize features
        df = self.normalize_features(df)
        
        # Step 3: Filter noise/outliers
        df = self.filter_noise(df)
        
        if len(df) == 0:
            print("Warning: No data remaining after noise filtering")
            return df
            
        # Step 4: Dimensionality reduction
        df = self.kaczmarz_sampling(df)
        
        # Step 5: Stratify tasks
        df = self.stratify_tasks(df)
        
        # NEW Step 6: Feature extraction
        df = self.extract_features(df)
        
        # Save processed data if output file specified
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\nProcessed data saved to {output_file}")
        
        return df
    
    def plot_resource_boxplots(self, df):
        if len(df) == 0:
            print("Warning: No data to plot")
            return
            
        plt.figure(figsize=(8, 6))
        
        # CPU Utilization
        if 'Resource_Type' in df.columns and 'Normalized_CPU_Utilization (%)' in df.columns:
            sns.boxplot(x='Resource_Type', y='Normalized_CPU_Utilization (%)', data=df)
            plt.title('CPU Utilization by Resource Type', fontweight="bold", fontsize=12)
            plt.xlabel('Resource Type', fontweight="bold", fontsize=10)
            plt.ylabel('Normalized CPU Utilization (%)', fontweight="bold", fontsize=10)
            plt.xticks(fontweight="bold")
            plt.yticks(fontweight="bold")
        else:
            plt.text(0.5, 0.5, 'No CPU data available', ha='center', va='center', fontweight="bold")
            
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(8, 6))
        
        # Memory Consumption
        if 'Resource_Type' in df.columns and 'Normalized_Memory_Consumption (MB)' in df.columns:
            sns.boxplot(x='Resource_Type', y='Normalized_Memory_Consumption (MB)', data=df)
            plt.title('Memory Consumption by Resource Type', fontweight="bold", fontsize=12)
            plt.xlabel('Resource Type', fontweight="bold", fontsize=10)
            plt.ylabel('Normalized Memory Consumption (MB)', fontweight="bold", fontsize=10)
            plt.xticks(fontweight="bold")
            plt.yticks(fontweight="bold")
        else:
            plt.text(0.5, 0.5, 'No Memory data available', ha='center', va='center', fontweight="bold")
        
        plt.tight_layout()
        plt.show()

    def detect_anomalies(self, df):
        """Anomaly Detection (Layer 3)"""
        print("\nStarting Anomaly Detection (Layer 3)")
        
        if len(df) == 0:
            print("Warning: Empty dataframe received for anomaly detection")
            return df
            
        # ALyFaO - Priority Inversion Detection
        df = self._detect_priority_inversions(df)
        
        # FeRoH - Envy-Freeness Cycle Detection
        df = self._detect_envy_cycles(df)
        
        # Falcon-inspired prioritization
        df = self._apply_falcon_prioritization(df)
        
        print("\nAnomaly detection completed")
        print("Detected anomalies summary:")
        print(df.filter(regex='ALyFaO_|FeRoH_|Falcon_').describe())
        return df
    
    def _detect_priority_inversions(self, df):
        if 'Normalized_Job_Priority' not in df.columns:
            df['Normalized_Job_Priority'] = df['Job_Priority'].map({
                'low': 0.3,
                'medium': 0.6,
                'high': 0.9
            })
        
        df['ALyFaO_Priority_Inversion_Risk'] = (
            0.4 * (df['Normalized_CPU_Utilization (%)'] < 0.2) +  
            0.3 * (df['Normalized_Memory_Consumption (MB)'] < 0.2) + 
            0.3 * df['Normalized_Job_Priority']
        )
        
        threshold = df['ALyFaO_Priority_Inversion_Risk'].quantile(0.85)
        df['ALyFaO_Priority_Inversion_Flag'] = df['ALyFaO_Priority_Inversion_Risk'] > threshold
        
        return df
    
    def _detect_envy_cycles(self, df):
        """FeRoH: Detect envy-freeness cycles using thermal clustering"""
        if len(df) == 0:
            return df
            
        if 'Normalized_Envy_Freeness_Score' in df.columns:
            df['FeRoH_Resource_Envy_Score'] = df['Normalized_Envy_Freeness_Score']
        elif ('Normalized_CPU_Utilization (%)' in df.columns and 
              'Normalized_Memory_Consumption (MB)' in df.columns and
              'WeiDiD_Node_Criticality' in df.columns):
            df['FeRoH_Resource_Envy_Score'] = (
                df['Normalized_CPU_Utilization (%)'] * 
                df['Normalized_Memory_Consumption (MB)'] * 
                (1 - df['WeiDiD_Node_Criticality'])
            )
        else:
            df['FeRoH_Resource_Envy_Score'] = np.nan
            
        if 'FeRoH_Resource_Envy_Score' in df.columns:
            threshold = np.percentile(df['FeRoH_Resource_Envy_Score'], 85)
            df['FeRoH_Envy_Cycle_Flag'] = df['FeRoH_Resource_Envy_Score'] > threshold
        else:
            df['FeRoH_Envy_Cycle_Flag'] = False
            
        return df
    
    def _apply_falcon_prioritization(self, df):
        """Falcon-inspired prioritization of tasks"""
        if len(df) == 0:
            return df
            
        if 'Normalized_Scheduling_Priority_Score' in df.columns:
            priority_score = df['Normalized_Scheduling_Priority_Score']
        elif 'WeiDiD_Node_Criticality' in df.columns:
            priority_score = df['WeiDiD_Node_Criticality']
        else:
            priority_score = 0.5  
            
        if 'WeiDiD_Delay_Likelihood' in df.columns and 'WeiDiD_Host_Affinity' in df.columns:
            df['Falcon_Priority_Score'] = (
                0.5 * priority_score +
                0.3 * df['WeiDiD_Delay_Likelihood'] +
                0.2 * df['WeiDiD_Host_Affinity']
            )
        else:
            df['Falcon_Priority_Score'] = priority_score
            
        conditions = [
            df['Falcon_Priority_Score'] > self.falcon_params['high_priority_threshold'],
            df['Falcon_Priority_Score'] < self.falcon_params['low_priority_threshold']
        ]
        choices = ['high', 'low']
        df['Falcon_Priority_Level'] = np.select(conditions, choices, default='medium')
        
        return df
    
    def optimize_placement(self, df):
        """Execution Layer (Layer 4): WhiSOT Placement Optimization"""
        print("\nStarting Placement Optimization (Layer 4)")
        
        if len(df) == 0:
            print("Warning: Empty dataframe received for placement optimization")
            return df
            
        # Tabu Search for affinity bias detection
        df = self._detect_affinity_bias(df)
        
        # White Shark Optimization for placement
        df = self._optimize_with_wso(df)
        
        print("\nPlacement optimization completed")
        return df
    
    def _detect_affinity_bias(self, df):
        """Tabu Search for affinity deadbolt bias detection"""
        if 'Actual_Host_ID' in df.columns:
            # Update tabu list with recently used hosts
            recent_hosts = df['Actual_Host_ID'].value_counts().nlargest(3).index
            self.tabu_list.extend(recent_hosts)
            
            df['WhiSOT_Affinity_Bias'] = df['Actual_Host_ID'].isin(self.tabu_list)
            
            df['WhiSOT_Bias_Score'] = df.groupby('Actual_Host_ID')['Actual_Host_ID'].transform('count') / len(df)
        else:
            df['WhiSOT_Affinity_Bias'] = False
            df['WhiSOT_Bias_Score'] = 0
            
        return df
    
    def _optimize_with_wso(self, df):
        """White Shark Optimization for placement"""
        if 'Actual_Host_ID' in df.columns:
            host_stats = df.groupby('Actual_Host_ID').agg({
                'Normalized_CPU_Utilization (%)': 'mean',
                'Normalized_Memory_Consumption (MB)': 'mean',
                'Normalized_PM_Utilization_Efficiency (%)': 'mean'
            }).reset_index()
            
            underutilized = host_stats[
                (~host_stats['Actual_Host_ID'].isin(self.tabu_list)) &
                (host_stats['Normalized_CPU_Utilization (%)'] < 0.5) &
                (host_stats['Normalized_Memory_Consumption (MB)'] < 0.5)
            ]
            
            if len(underutilized) > 0:
                target_host = underutilized.iloc[
                    (underutilized['Normalized_CPU_Utilization (%)'] + 
                     underutilized['Normalized_Memory_Consumption (MB)']).argmin()
                ]['Actual_Host_ID']
            else:
                target_host = "Optimal_PM"
        else:
            target_host = "Optimal_PM"
        
        df['WhiSOT_Suggested_Host'] = np.where(
            df['WhiSOT_Affinity_Bias'],
            target_host,  
            "Optimal_PM"  
        )
        
        return df
    
    def monitor_and_control(self, df):
        """Control Layer (Layer 5): Real-time Monitoring"""
        print("\nStarting Real-time Monitoring (Layer 5)")
        
        if len(df) == 0:
            print("Warning: Empty dataframe received for monitoring")
            return df
            
        # VM Health Monitoring
        df = self._monitor_vm_health(df)
        
        # Context-Switch Monitoring
        df = self._monitor_context_switches(df)
        
        # Feedback to upper layers
        df = self._update_feedback_loop(df)
        
        print("\nMonitoring completed")
        return df
    
    def _monitor_vm_health(self, df):
        """Track VM resource usage and health"""
        # CPU monitoring
        if 'Normalized_CPU_Utilization (%)' in df.columns:
            cpu_threshold = 0.9  
            df['Control_CPU_Overload'] = df['Normalized_CPU_Utilization (%)'] > cpu_threshold
        else:
            df['Control_CPU_Overload'] = False
            
        # Memory monitoring
        if 'Normalized_Memory_Consumption (MB)' in df.columns:
            mem_threshold = 0.85  
            df['Control_Memory_Overload'] = df['Normalized_Memory_Consumption (MB)'] > mem_threshold
        else:
            df['Control_Memory_Overload'] = False
            
        # Network monitoring
        if 'Normalized_Network_Bandwidth_Utilization (Mbps)' in df.columns:
            net_threshold = 0.8  
            df['Control_Network_Congestion'] = df['Normalized_Network_Bandwidth_Utilization (Mbps)'] > net_threshold
        else:
            df['Control_Network_Congestion'] = False
            
        # Overall health score
        health_components = []
        if 'Normalized_CPU_Utilization (%)' in df.columns:
            health_components.append(0.4 * (1 - df['Normalized_CPU_Utilization (%)']))
        if 'Normalized_Memory_Consumption (MB)' in df.columns:
            health_components.append(0.3 * (1 - df['Normalized_Memory_Consumption (MB)']))
        if 'Normalized_Network_Bandwidth_Utilization (Mbps)' in df.columns:
            health_components.append(0.3 * (1 - df['Normalized_Network_Bandwidth_Utilization (Mbps)']))
            
        if health_components:
            df['Control_VM_Health_Score'] = sum(health_components) / len(health_components)
        else:
            df['Control_VM_Health_Score'] = 0.5  
            
        return df
    
    def _monitor_context_switches(self, df):
        if 'Normalized_Predicted_Context_Switch_Penalty (ms)' in df.columns:
            cs_threshold = np.percentile(df['Normalized_Predicted_Context_Switch_Penalty (ms)'], 90)
            df['Control_CS_Penalty_Flag'] = df['Normalized_Predicted_Context_Switch_Penalty (ms)'] > cs_threshold
            df['Control_CS_Penalty_Score'] = (
                df['Normalized_Predicted_Context_Switch_Penalty (ms)'] / cs_threshold
            ).clip(upper=1.0)
        elif 'Normalized_Task_Waiting_Time (ms)' in df.columns:
            cs_threshold = np.percentile(df['Normalized_Task_Waiting_Time (ms)'], 90)
            df['Control_CS_Penalty_Flag'] = df['Normalized_Task_Waiting_Time (ms)'] > cs_threshold
            df['Control_CS_Penalty_Score'] = (
                df['Normalized_Task_Waiting_Time (ms)'] / cs_threshold
            ).clip(upper=1.0)
        else:
            df['Control_CS_Penalty_Flag'] = False
            df['Control_CS_Penalty_Score'] = 0
            
        return df
    
    def _update_feedback_loop(self, df):
        # Update WeiDiD parameters based on observed delays
        if 'Control_CS_Penalty_Flag' in df.columns and 'Normalized_Task_Waiting_Time (ms)' in df.columns:
            high_delay_tasks = df[df['Control_CS_Penalty_Flag']]
            if len(high_delay_tasks) > 0:
                avg_delay = high_delay_tasks['Normalized_Task_Waiting_Time (ms)'].mean()
                self.stratification_rules['short_task']['max_duration'] = min(
                    500, 
                    self.stratification_rules['short_task']['max_duration'] * (1 + avg_delay)
                )
        
        if 'ALyFaO_Priority_Inversion_Flag' in df.columns:
            inversion_cases = df[df['ALyFaO_Priority_Inversion_Flag']]
            if len(inversion_cases) > 0:
                self.falcon_params['high_priority_threshold'] = max(
                    0.7,
                    self.falcon_params['high_priority_threshold'] - 0.05
                )
        
        return df
    
    def full_pipeline(self, input_data, output_file=None):
        if isinstance(input_data, str):
            # Step 1-6: Existing preprocessing from file
            df = self.preprocess(input_data)
        elif isinstance(input_data, pd.DataFrame):
            # Start with provided DataFrame
            df = input_data.copy()
            
            # Step 2: Normalize features
            df = self.normalize_features(df)
            
            # Step 3: Filter noise/outliers
            df = self.filter_noise(df)
            
            if len(df) == 0:
                print("Warning: No data remaining after noise filtering")
                return df
                
            # Step 4: Dimensionality reduction
            df = self.kaczmarz_sampling(df)
            
            # Step 5: Stratify tasks
            df = self.stratify_tasks(df)
            
            # Step 6: Feature extraction
            df = self.extract_features(df)
        else:
            raise ValueError("Input must be either a file path or DataFrame")
        
        if len(df) == 0:
            print("Warning: No data remaining after preprocessing")
            return df
            
        # Step 7: Anomaly Detection (Layer 3)
        df = self.detect_anomalies(df)
        
        # Step 8: Placement Optimization (Layer 4)
        df = self.optimize_placement(df)
        
        # Step 9: Monitoring & Control (Layer 5)
        df = self.monitor_and_control(df)
        
        # Save final output
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\nFinal processed data saved to {output_file}")
        
        return df
    
class OPenCMInterface:
    def __init__(self, reference_data_path=None):
        self.profiler = TaskProfilerEngine()
        self.reference_data = None
        
        if reference_data_path:
            try:
                self.reference_data = self.profiler.load_data(reference_data_path)
                
                print("\nFitting scaler on reference data...")
                self.profiler.normalize_features(self.reference_data.copy())
                
                print(f"Loaded and pre-processed reference data with {len(self.reference_data)} entries")
            except Exception as e:
                print(f"Warning: Could not load reference data - {str(e)}")
        
        self.input_mapping = {
            'Job_ID': 'Job_ID',
            'VM_Total_Memory_Capacity (MB)': 'VM_Total_Memory_Capacity (MB)',
            'Predicted_Execution_Duration (ms)': 'Predicted_Execution_Duration (ms)',
            'Task_Type': 'Task_Type',
            'CPU_Utilization (%)': 'CPU_Utilization (%)',
            'Memory_Consumption (MB)': 'Memory_Consumption (MB)',
            'Network_Bandwidth_Utilization (Mbps)': 'Network_Bandwidth_Utilization (Mbps)',
            'Job_Priority': 'Job_Priority',
            'Scheduling_Priority_Score': 'Scheduling_Priority_Score',
            'Task_Start_Time': 'Task_Start_Time',
            'Task_End_Time': 'Task_End_Time',
            'System_Throughput (tasks/sec)': 'System_Throughput (tasks/sec)',
            'Task_Waiting_Time (ms)': 'Task_Waiting_Time (ms)',
            'Error_Rate (%)': 'Error_Rate (%)',
            'Task_Execution_Time (ms)': 'Task_Execution_Time (ms)',
            'Context_Switch_Overhead (%)': 'Context_Switch_Overhead (%)',
            'Dominant_Resource_Share (%)': 'Dominant_Resource_Share (%)'
        }
        
        self.output_mapping = {
            col: col for col in [
                'Job_ID', 'Task_Start_Time', 'Task_End_Time', 
                'CPU_Utilization (%)', 'Memory_Consumption (MB)',
                'Task_Execution_Time (ms)', 'System_Throughput (tasks/sec)',
                'Task_Waiting_Time (ms)', 'Network_Bandwidth_Utilization (Mbps)',
                'Job_Priority', 'Error_Rate (%)', 'Predicted_Execution_Duration (ms)',
                'Context_Switch_Overhead (%)', 'Dominant_Resource_Share (%)'
            ]
        }
        
    def validate_input(self, input_data):
        if not isinstance(input_data, dict) or 'Job_ID' not in input_data:
            return {"valid": False, "message": "Input must be a dictionary with Job_ID"}
        
        if self.reference_data is None:
            return {"valid": True, "message": "No reference data for validation"}
        
        job_id = input_data['Job_ID']
        reference_row = self.reference_data[self.reference_data['Job_ID'] == job_id]
        
        if len(reference_row) == 0:
            return {"valid": False, "message": f"Job_ID {job_id} not found"}
        
        validation_results = []
        reference_values = reference_row.iloc[0].to_dict()
        
        for field, input_value in input_data.items():
            if field in reference_values:
                ref_value = reference_values[field]
                
                if field in ['Task_Start_Time', 'Task_End_Time']:
                    try:
                        input_dt = pd.to_datetime(input_value)
                        ref_dt = pd.to_datetime(ref_value)
                        
                        if input_dt != ref_dt:
                            validation_results.append({
                                'field': field,
                                'input_value': input_value,
                                'reference_value': ref_value,
                                'match': False
                            })
                    except:
                        if str(input_value).strip() != str(ref_value).strip():
                            validation_results.append({
                                'field': field,
                                'input_value': input_value,
                                'reference_value': ref_value,
                                'match': False
                            })
                elif isinstance(input_value, (int, float)) and isinstance(ref_value, (int, float)):
                    if not np.isclose(input_value, ref_value, rtol=0.01):
                        validation_results.append({
                            'field': field,
                            'input_value': input_value,
                            'reference_value': ref_value,
                            'match': False
                        })
                elif str(input_value).strip().lower() != str(ref_value).strip().lower():
                    validation_results.append({
                        'field': field,
                        'input_value': input_value,
                        'reference_value': ref_value,
                        'match': False
                    })
        
        if not validation_results:
            return {"valid": True, "message": "All values match reference data"}
        else:
            return {
                "valid": False,
                "message": f"{len(validation_results)} mismatches found",
                "mismatches": validation_results
            }
    
    def transform_input(self, user_input):
        if not isinstance(user_input, dict):
            raise ValueError("Input must be a dictionary")
            
        task_data = {}
        
        for user_field, dataset_col in self.input_mapping.items():
            if user_field in user_input:
                value = user_input[user_field]
                if isinstance(value, (int, float)):
                    task_data[dataset_col] = float(value)
                else:
                    task_data[dataset_col] = value
        
        if isinstance(user_input.get('Resource_Requirement'), dict):
            resources = user_input['Resource_Requirement']
            for res_type, col_suffix in [('CPU', 'CPU_Utilization (%)'),
                                        ('RAM', 'Memory_Consumption (MB)'),
                                        ('Network', 'Network_Bandwidth_Utilization (Mbps)')]:
                if res_type in resources:
                    task_data[col_suffix] = float(resources[res_type])
        
        if 'Dominant_Resource_Share (%)' not in task_data:
            if 'CPU_Utilization (%)' in task_data and 'Memory_Utilization (%)' in task_data:
                task_data['Dominant_Resource_Share (%)'] = max(
                    task_data['CPU_Utilization (%)'],
                    task_data['Memory_Utilization (%)']
                )
            elif 'CPU_Utilization (%)' in task_data:
                task_data['Dominant_Resource_Share (%)'] = task_data['CPU_Utilization (%)']
            elif 'Memory_Utilization (%)' in task_data:
                task_data['Dominant_Resource_Share (%)'] = task_data['Memory_Utilization (%)']
            else:
                task_data['Dominant_Resource_Share (%)'] = 0

        if 'Task_Start_Time' not in task_data:
            task_data['Task_Start_Time'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        if 'Task_End_Time' not in task_data:
            task_data['Task_End_Time'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        if 'Error_Rate (%)' not in task_data:
            task_data['Error_Rate (%)'] = 0.0
        
        return task_data

    def transform_output(self, processed_df):
        if len(processed_df) == 0:
            return {"error": "No data available"}
        
        row = processed_df.iloc[0].to_dict()
        output = {}
        
        for dataset_col, output_field in self.output_mapping.items():
            if dataset_col in row and pd.notna(row[dataset_col]):
                output[output_field] = row[dataset_col]
            elif f"Normalized_{dataset_col}" in row and pd.notna(row[f"Normalized_{dataset_col}"]):
                if hasattr(self.profiler.scaler, 'scale_'):
                    try:
                        idx = list(self.profiler.scaler.feature_names_in_).index(dataset_col)
                        value = row[f"Normalized_{dataset_col}"] * self.profiler.scaler.scale_[idx] + self.profiler.scaler.min_[idx]
                        output[output_field] = value
                    except (ValueError, IndexError):
                        output[output_field] = row[f"Normalized_{dataset_col}"]
                else:
                    output[output_field] = row[f"Normalized_{dataset_col}"]
        
        output['Load_Monitoring'] = {
            'CPU_Usage': f"{output.get('CPU_Utilization (%)', 0):.1f}%",
            'Memory_Usage': f"{output.get('Memory_Consumption (MB)', 0)/1024:.2f}GB",
            'Network_Usage': f"{output.get('Network_Bandwidth_Utilization (Mbps)', 0):.2f}Mbps",
            'Context_Switch_Overhead': f"{output.get('Context_Switch_Overhead (%)', 0):.1f}%"
        }
        
        output['Feedback_Log'] = self._generate_feedback_log(row)
        output['Self_Learning_Update'] = self._generate_self_learning_update(row)
        output['Layer_5_Decision'] = self._generate_layer5_decision(row)
        
        return output

    def _generate_feedback_log(self, row):
        feedback = {
            "Status": "OK",
            "Details": {
                "Issue": "",
                "Expected": "",
                "Actual": "",
                "Action": ""
            }
        }
        
        if row.get('CPU_Utilization (%)', 0) > 85:  
            feedback["Status"] = "Warning"  
            feedback["Details"] = {
                "Issue": "High CPU utilization",
                "Expected": "<85%",
                "Actual": f"{row['CPU_Utilization (%)']:.1f}%",
                "Action": "Monitor closely"
            }
        
        if 'Predicted_Execution_Duration (ms)' in row and 'Task_Execution_Time (ms)' in row:
            if row['Predicted_Execution_Duration (ms)'] > 0:  
                deviation = (row['Task_Execution_Time (ms)'] - row['Predicted_Execution_Duration (ms)']) / row['Predicted_Execution_Duration (ms)']
                if deviation > 0.15:  
                    feedback["Status"] = "Warning"
                    feedback["Details"] = {
                        "Issue": "Execution time deviation",
                        "Expected": f"<{row['Predicted_Execution_Duration (ms)']*1.15:.0f}ms",
                        "Actual": f"{row['Task_Execution_Time (ms)']:.0f}ms",
                        "Action": "Check for resource contention"
                    }
        
        return feedback

    def _generate_self_learning_update(self, row):
        update = {
            "Status": "OK",
            "Triggered_Modules": []
        }
        
        if row.get('ALyFaO_Priority_Inversion_Flag', False):
            update["Status"] = "Error"
            update["Triggered_Modules"].append({
                "Layer": "Layer_3",
                "Module": "ALyFaO",
                "Reason": "Priority inversion detected"
            })
        
        if row.get('FeRoH_Envy_Cycle_Flag', False):
            update["Status"] = "Error"
            update["Triggered_Modules"].append({
                "Layer": "Layer_3",
                "Module": "FeRoH",
                "Reason": "Resource envy detected"
            })
        
        return update

    def _generate_layer5_decision(self, row):
        decision = {
            "Stop_Layer_5": False,
            "Next_Layer_Routing": [],
            "Reason": "Normal operation"
        }
        
        feedback = self._generate_feedback_log(row)
        learning = self._generate_self_learning_update(row)
        
        if feedback["Status"] == "Error":
            decision["Next_Layer_Routing"].append("Layer_2")
            decision["Reason"] = "Feedback error"
        
        if learning["Status"] == "Error":
            decision["Next_Layer_Routing"].append("Layer_3")
            decision["Reason"] += " and Learning error" if decision["Reason"] != "Normal operation" else "Learning error"
        
        if not decision["Next_Layer_Routing"]:
            decision["Stop_Layer_5"] = True
        
        return decision

    def process_task(self, input_json):
        try:
            validation = self.validate_input(input_json)
            if not validation['valid']:
                return {
                    "status": "validation_error",
                    "message": validation['message'],
                    "details": validation.get('mismatches', [])
                }
            
            dataset_format = self.transform_input(input_json)
            input_df = pd.DataFrame([dataset_format])
            
            if len(input_df) == 1:
                print("\nProcessing single task with reference scaler...")
                
                if hasattr(self.profiler.scaler, 'scale_'):
                    reference_cols = list(self.profiler.scaler.feature_names_in_)
                    
                    full_input_df = pd.DataFrame(columns=reference_cols)
                    for col in reference_cols:
                        if col in input_df.columns:
                            full_input_df[col] = input_df[col]
                        else:
                            full_input_df[col] = 0.0  
                    
                    normalized_values = (full_input_df[reference_cols] - self.profiler.scaler.min_) / self.profiler.scaler.scale_
                    
                    for col in reference_cols:
                        input_df[f"Normalized_{col}"] = normalized_values[col]
                
                filtered_df = input_df
            else:
                input_df = self.profiler.normalize_features(input_df)
                filtered_df = self.profiler.filter_noise(input_df)
            
            if len(filtered_df) == 0:
                return {
                    "status": "error",
                    "message": "No data remaining after processing",
                    "input_received": str(input_json)
                }
            
            processed_df = filtered_df.copy()
            processed_df = self.profiler.kaczmarz_sampling(processed_df)
            processed_df = self.profiler.stratify_tasks(processed_df)
            processed_df = self.profiler.extract_features(processed_df)
            processed_df = self.profiler.detect_anomalies(processed_df)
            processed_df = self.profiler.optimize_placement(processed_df)
            processed_df = self.profiler.monitor_and_control(processed_df)
            
            output = self.transform_output(processed_df)
            
            return {
                "status": "success",
                "data": output
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "input_received": str(input_json)
            }

# Example usage
if __name__ == "__main__":
    lirakat = TaskProfilerEngine()
    
    input_csv = r".\dataset\Cloud_Model_Dataset.csv"
    
    raw_data = pd.read_csv(input_csv)
    processed_data = lirakat.preprocess(input_csv)
    
    metrics = ["CPU_Utilization (%)", "Memory_Consumption (MB)", "Task_Execution_Time (ms)", 
               "System_Throughput (tasks/sec)", "Task_Waiting_Time (ms)", "Energy_Efficiency_Index"]
    plt.figure(figsize=(10,7))
    corr = raw_data[metrics].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Key Metrics", fontsize=12, fontweight="bold")
    plt.xticks(fontsize=10, fontweight="bold")
    plt.yticks(fontsize=10, fontweight="bold")
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(processed_data.index, processed_data['Normalized_CPU_Utilization (%)'], label='CPU', color='orange')
    plt.plot(processed_data.index, processed_data['Normalized_Memory_Consumption (MB)'], label='Memory', color='blue')
    plt.xlabel('Task Index', fontsize=10, fontweight="bold")
    plt.ylabel('Normalized Value', fontsize=10, fontweight="bold")
    plt.title('CPU & Memory Utilization Over Tasks', fontsize=12, fontweight="bold")
    plt.xticks(fontsize=10, fontweight="bold")
    plt.yticks(fontsize=10, fontweight="bold")
    plt.legend(prop={'weight': 'bold'}, title_fontsize='10', title='Legend')
    plt.show()
    
    print("\nSample of processed data:")
    print(processed_data.head())
    
    print("\nExtracted WeiDiD features:")
    print(processed_data.filter(regex='WeiDiD_').head())

    lirakat.plot_resource_boxplots(processed_data)
                
    final_data = lirakat.full_pipeline(input_csv)
    
    print("\nMonitoring alerts summary:")
    print(f"CPU Overload cases: {final_data['Control_CPU_Overload'].sum()}")
    print(f"Memory Overload cases: {final_data['Control_Memory_Overload'].sum()}")
    print(f"Network Congestion cases: {final_data['Control_Network_Congestion'].sum()}")
    print(f"Context Switch Penalties: {final_data['Control_CS_Penalty_Flag'].sum()}")
    
    plt.figure(figsize=(12, 6))
    if all(col in processed_data.columns for col in ['Normalized_CPU_Utilization (%)', 
                                                   'Normalized_Memory_Consumption (MB)',
                                                   'Normalized_Network_Bandwidth_Utilization (Mbps)']):
        sample = processed_data.sample(min(50, len(processed_data))) 
        sample = sample.sort_values('Normalized_Task_Execution_Time (ms)')
        
        plt.bar(range(len(sample)), sample['Normalized_CPU_Utilization (%)'], 
               label='CPU', color='orange')
        plt.bar(range(len(sample)), sample['Normalized_Memory_Consumption (MB)'], 
               bottom=sample['Normalized_CPU_Utilization (%)'], 
               label='Memory', color='blue')
        plt.bar(range(len(sample)), sample['Normalized_Network_Bandwidth_Utilization (Mbps)'], 
               bottom=sample['Normalized_CPU_Utilization (%)'] + sample['Normalized_Memory_Consumption (MB)'], 
               label='Network', color='green')
        
        plt.xlabel('Task Sample', fontsize=10, fontweight="bold")
        plt.ylabel('Normalized Resource Usage', fontsize=10, fontweight="bold")
        plt.title('Stacked Resource Usage per Task', fontsize=12, fontweight="bold")
        plt.xticks(fontsize=10, fontweight="bold")
        plt.yticks(fontsize=10, fontweight="bold")
        plt.legend(prop={'weight': 'bold'}, title_fontsize='10', title='Legend')
    else:
        plt.text(0.5, 0.5, 'Missing resource data', ha='center', va='center')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    sns.histplot(raw_data["VM_Placement_Diversity_Score"], kde=True, bins=15)
    plt.title("Distribution of VM Placement Diversity Score", fontsize=12, fontweight="bold")
    plt.xlabel("VM Placement Diversity Score", fontsize=10, fontweight="bold")
    plt.ylabel("Frequency", fontsize=10, fontweight="bold")
    plt.xticks(fontsize=10, fontweight="bold")
    plt.yticks(fontsize=10, fontweight="bold")
    plt.show()
    
    plt.figure(figsize=(6,4))
    sns.histplot(raw_data["Task_Execution_Time (ms)"], kde=True, bins=15, color="orange")
    plt.title("Task Execution Time Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Execution Time (ms)", fontsize=10, fontweight="bold")
    plt.ylabel("Frequency", fontsize=10, fontweight="bold")
    plt.xticks(fontsize=10, fontweight="bold")
    plt.yticks(fontsize=10, fontweight="bold")
    plt.show()
        
    opencm = OPenCMInterface(input_csv)

    task_request = {
    "Job_ID": "JOB_1",
    "VM_Total_Memory_Capacity (MB)": 16384,
    "Predicted_Execution_Duration (ms)": 2522,
    "Task_Type": "CPU-Bound",
    "CPU_Utilization (%)": 39.96,
    "Memory_Consumption (MB)": 3622,
    "Network_Bandwidth_Utilization (Mbps)": 112.97,
    "Job_Priority": "Low",
    "Scheduling_Priority_Score": 0.486,
    "Task_Start_Time": "01-01-2024  00:00:00",
    "Task_End_Time": "01-01-2024  00:00:00",
    "Error_Rate (%)": 1.65,
    "Task_Execution_Time (ms)": 2734,
    "Context_Switch_Overhead (%)": 0.98,
    "Dominant_Resource_Share (%)": 65.68  
    }

    try:
       result = opencm.process_task(task_request)
       
       print("OPen-CM Processing Result:")
       print(json.dumps(result, indent=2))
    except Exception as e:
       print(f"Error processing task: {str(e)}")
       print(f"Task request structure: {task_request}")