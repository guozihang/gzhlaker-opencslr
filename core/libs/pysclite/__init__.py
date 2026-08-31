"""pysclite: a Python port of NIST SCTK's sclite speech scoring tool.

Replicates sclite's DP alignment, statistics and report formats for
TRN/CTM/STM inputs.  All internal text is handled as bytes, mirroring
the C implementation's unsigned-char semantics.
"""

__version__ = '2.10'
