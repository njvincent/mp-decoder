# Time-scan result organization

Results are grouped by the update-time and cleanup-time multiples encoded in
each filename:

- `Fu` is the update-time multiple of `T`.
- `Fc` is the cleanup-time multiple of `T`.
- `Tu = Fu * T` and `Tc = Fc * T`, with the filename's `L` value serving as
  `T` for these runs.

| Directory | Update time | Cleanup time |
| --- | ---: | ---: |
| `update_1T_cleanup_2T/` | `1T` | `2T` |
| `update_1T_cleanup_4T/` | `1T` | `4T` |
| `update_2T_cleanup_2T/` | `2T` | `2T` |
| `update_2T_cleanup_4T/` | `2T` | `4T` |
| `update_3T_cleanup_2T/` | `3T` | `2T` |
| `update_3T_cleanup_4T/` | `3T` | `4T` |

