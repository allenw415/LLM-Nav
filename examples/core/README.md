# Core Examples

Focused commands for individual navigation capabilities:

```bash
python3 -m sa_nav.cli.parse_instruction --instruction "Find the way from Room 8 to Room 23."
python3 -m sa_nav.cli.resolve_source_pano --source-room-id "Room 8"
python3 -m sa_nav.cli.plan_room_route --source-room-id "Room 8" --target-room-id "Room 23"
python3 -m sa_nav.cli.run_pano_perception --pano-id "7grGsbOXqpEMDLgTG6VfmQ"
python3 -m sa_nav.cli.run_localization --mode synthetic --json
```

The CLI adapters live in `sa_nav.cli`; navigation logic stays in the runtime
packages.
