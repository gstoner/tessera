# 10.6. Image Sampler

> RDNA4 ISA — pages 137–137

Bits      Size       Name                 Comments
183       1          corner samples       Describes how texels were generated in the resource. 0=center sampled, 1 = corner
                     mod                  sampled.
198:186   13         min_lod              smallest LOD allowed for PRTs, U5.8 format

A resource that is all zeros is treated as 'unbound': it returns all zeros and not generate a memory transaction.
The "resource-level" field is ignored when checking for "all zeros".

10.6. Image Sampler
The sampler resource (also referred to as S#) defines what operations to perform on texture map data loaded
by sample instructions. These are primarily address clamping and filter options. Sampler resources are
defined in four consecutive SGPRs and are supplied to the texture cache with every sample instruction.

                                               Table 61. Image Sampler Definition
Bits          Size      Name                         Description
2:0           3         clamp x                      Clamp/wrap mode:
                                                     0: Wrap
                                                     1: Mirror
5:3           3         clamp y                      2: ClampLastTexel
                                                     3: MirrorOnceLastTexel
                                                     4: ClampHalfBorder
8:6           3         clamp z                      5: MirrorOnceHalfBorder
                                                     6: ClampBorder
                                                     7: MirrorOnceBorder
11:9          3         max aniso ratio              0 = 1:1
                                                     1 = 2:1
                                                     2 = 4:1
                                                     3 = 8:1
                                                     4 = 16:1
14:12         3         depth compare func           0: Never
                                                     1: Less
                                                     2: Equal
                                                     3: Less than or equal
                                                     4: Greater
                                                     5: Not equal
                                                     6: Greater than or equal
                                                     7: Always
15            1         force unnormalized           Force address cords to be unorm: 0 = address coordinates are
                                                     normalized, in [0,1); 1 = address coordinates are unnormalized in the
                                                     range [0,dim).
18:16         3         aniso threshold              threshold under which floor(aniso ratio) determines number of samples
                                                     and step size
19            1         mc coord trunc               enables bilinear blend fraction truncation to 1 bit for motion
                                                     compensation
20            1         force degamma                force format to srgb if data_format allows
26:21         6         aniso bias                   6 bits, in u1.5 format.
27            1         trunc coord                  selects texel coordinate rounding or truncation.
28            1         disable cube wrap            disables seamless DX10 cubemaps, allows cubemaps to clamp according
                                                     to clamp_x and clamp_y fields
30:29         2         filter_mode                  0 = Blend (lerp); 1 = min, 2 = max.
