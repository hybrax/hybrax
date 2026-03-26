# Some notes

## bpbench QoL changes
- When defining a medium omitting a component should be treated as zero concentration for that component.
- We might want to rename the `FeedMedium` dataclass as it contains feed and base and potentially other medium types and the current name could confuse people. Maybe use something like "AddedMedium" or similar.
- Do we really need `ReactorMedium` and `FeedMedium` and `ReactorMediumComponent` and `FeedMediumComponent`? I realise that reactor medium components are measured and feed medium components are not, but can't this distinction is already crystallized in `process.reactor_medium` vs `process.volume.volume_changes` anyway. I think we could simplify the API here.