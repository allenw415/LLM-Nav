# Flow Examples

End-to-end command for the complete navigation flow:

```bash
python3 -m st_nav.cli.run_navigation --instruction "Find the way from Room 8 to Room 23."
```

This parses an instruction, resolves the source panorama, runs the episode loop,
and writes navigation traces.
