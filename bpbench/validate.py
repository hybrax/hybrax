# TODO for later: add validation in separate validate.py module
    # def validate_volume_consistency(self, time_axis: Optional[TimeAxis] = None, 
    #                                final_volume: Optional[float] = None) -> tuple[bool, str]:
    #     """
    #     Validate that volume changes sum to expected final volume.
        
    #     Returns:
    #         (is_valid, message): Tuple of validation result and descriptive message
    #     """
    #     if self.initial_volume is None:
    #         return (True, "No initial volume specified, skipping validation")
        
    #     if not self.volume_changes:
    #         return (True, "No volume changes to validate")
        
    #     # Calculate total volume change
    #     total_change = 0.0
    #     messages = []
        
    #     for name, change in self.volume_changes.items():
    #         if change.continuous and change.timeseries is not None:
    #             # For continuous changes, check if data is cumulative or rate
    #             if change.timeseries.raw is not None:
    #                 times = change.timeseries.raw.timepoints
    #                 values = change.timeseries.raw.values
                    
    #                 if len(times) > 1:
    #                     # Check unit to determine if cumulative or rate
    #                     if change.unit == "L" or change.unit == self.volume_unit:
    #                         # Cumulative volume: final - initial
    #                         change_vol = float(values[-1] - values[0])
    #                     elif "/" in change.unit:
    #                         # Rate (e.g., "L/h"): integrate using trapezoidal rule
    #                         dt = jnp.diff(times)
    #                         avg_rates = (values[:-1] + values[1:]) / 2.0
    #                         change_vol = float(jnp.sum(dt * avg_rates))
    #                     else:
    #                         # Unknown unit, assume cumulative
    #                         change_vol = float(values[-1] - values[0])
                        
    #                     total_change += change_vol
    #                     messages.append(f"  {name}: +{change_vol:.2f} {self.volume_unit} (continuous)")
    #         elif not change.continuous and change.values is not None:
    #             # For discrete changes, sum all values
    #             change_vol = float(jnp.sum(change.values))
    #             total_change += change_vol
    #             messages.append(f"  {name}: +{change_vol:.2f} {self.volume_unit} (discrete)")
        
    #     calculated_final = self.initial_volume + total_change
        
    #     if final_volume is not None:
    #         diff = abs(calculated_final - final_volume)
    #         rel_diff = diff / final_volume if final_volume > 0 else 0
            
    #         messages.insert(0, f"Initial volume: {self.initial_volume:.2f} {self.volume_unit}")
    #         messages.append(f"Total change: {total_change:.2f} {self.volume_unit}")
    #         messages.append(f"Calculated final: {calculated_final:.2f} {self.volume_unit}")
    #         messages.append(f"Expected final: {final_volume:.2f} {self.volume_unit}")
    #         messages.append(f"Difference: {diff:.2f} {self.volume_unit} ({rel_diff*100:.1f}%)")
            
    #         if rel_diff > 0.05:  # More than 5% difference
    #             return (False, "Volume inconsistency detected:\n" + "\n".join(messages))
    #         else:
    #             return (True, "Volume balance OK:\n" + "\n".join(messages))
    #     else:
    #         messages.insert(0, f"Initial volume: {self.initial_volume:.2f} {self.volume_unit}")
    #         messages.append(f"Calculated final: {calculated_final:.2f} {self.volume_unit}")
    #         return (True, "Volume changes calculated:\n" + "\n".join(messages))
    
    # def validate_feed_components(self, process_feeds: Dict[str, Feed], 
    #                              dynamic_variables: Dict[str, TimeSeries]) -> tuple[bool, str]:
    #     """
    #     Validate that feed compositions are properly defined for volume changes.
        
    #     For each VolumeChange with a feed_medium reference:
    #     - The referenced feed must exist in process_feeds
    #     - Warning if feed components don't cover all dynamic variables
        
    #     Args:
    #         process_feeds: Dictionary of Feed objects from Process.feeds
    #         dynamic_variables: Dictionary of TimeSeries from Process.dynamic_variables
            
    #     Returns:
    #         (is_valid, message): Tuple of validation result and descriptive message
    #     """
    #     messages = []
    #     all_valid = True
        
    #     for vc_name, vc in self.volume_changes.items():
    #         # Check if this volume change has a feed
    #         feed = None
    #         if vc.feed_medium is not None:
    #             # Reference to Process.feeds
    #             if vc.feed_medium not in process_feeds:
    #                 messages.append(f"ERROR: VolumeChange '{vc_name}' references feed '{vc.feed_medium}' "
    #                               f"which is not defined in Process.feeds")
    #                 all_valid = False
    #             else:
    #                 feed = process_feeds[vc.feed_medium]
    #         elif vc.feed is not None:
    #             # Inline feed definition
    #             feed = vc.feed
            
    #         # If there's a feed, validate component coverage
    #         if feed is not None:
    #             missing_components = []
    #             for var_name in dynamic_variables.keys():
    #                 if var_name not in feed.components:
    #                     missing_components.append(var_name)
                
    #             if missing_components:
    #                 messages.append(f"WARNING: VolumeChange '{vc_name}' feed '{feed.name}' "
    #                               f"is missing concentrations for dynamic variables: {missing_components}")
        
    #     if not messages:
    #         return (True, "All feed components properly defined")
        
    #     return (all_valid, "\n".join(messages))