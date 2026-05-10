| id | name | operation | result |
|----|------|-----------|--------|
| 17 | pretrained.decoder.input_latent_bottleneck.0.bias | roll to the right | produce more harmonic and brighter timbres |
| 17 | pretrained.decoder.input_latent_bottleneck.0.bias | squeeze | sharper, more compressed sound (a bit like OTF) |
| 24 | pretrained.decoder.input_latent_bottleneck.6.weight | roll to the right | smear the transients |
| 26 | pretrained.decoder.input_latent_bottleneck.7.weight | scale | low – lower volume, higher dynamics; mid – introducing distortion; high – introducing extra tonal timbral elements |
| 32 | pretrained.decoder.input_features_bottleneck.3.weight | various | various operations embellish different parts of the spectrum in a difficult to explain way |
| 34 | pretrained.decoder.input_features_bottleneck.4.weight | flip | add a lot of highs |
| 36 | pretrained.decoder.input_features_bottleneck.6.weight | draw | various interesting short-term timbral glitches |
| 36 | pretrained.decoder.input_features_bottleneck.6.weight | roll | effects similar to highpass / lowpass filtering |
| 37 | pretrained.decoder.input_features_bottleneck.6.weight | offset up | bring up brightness and detail |
| 40 | pretrained.decoder.gru.weight_ih_l0 | roll | abrupt timbral changes, from slight smearing to complete deconstruction |
| 40 | pretrained.decoder.gru.weight_ih_l0 | scale | lower values – rhythmic modulation; higher values – distortion |
| 40 | pretrained.decoder.gru.weight_ih_l0 | drawing | drawing in area around 75–90% of the array brings timbral changes without affecting temporal structure |
| 41 | pretrained.decoder.gru.weight_hh_l0 | scale | expanding – interesting array of temporal glitches; contracting – timbral shift |
| 41 | pretrained.decoder.gru.weight_hh_l0 | drawing | lower half – rhythmic modulation; upper half – distortion; overall – a spectrum of glitches, timbral shifts and modulations |
| 44 | pretrained.decoder.gru.weight_ih_l1 | scaling | contracting – fast rhythmic modulation |
| 44 | pretrained.decoder.gru.weight_ih_l1 | rolling | all kinds of timbral shifts |
| 59 | pretrained.decoder.inter_mlp.7.bias | scaling, offseting | low pass filtering |
| 58 | pretrained.decoder.inter_mlp.7.weight | offseting | upwards – adding noise and reverb; downwards – spectral modeling style effects |
| 52 | pretrained.decoder.inter_mlp.3.weight | rolling | small room reverbs and various distortions |
| 52 | pretrained.decoder.inter_mlp.3.weight | drawing | smearing and drone |
| 60 | pretrained.decoder.output_params.weight | rolling | spectral modeling style effects |
| 60 | pretrained.decoder.output_params.weight | scaling | scaling up – sinusoidal modeling, watery sounds |
| 60 | pretrained.decoder.output_params.weight | drawing | adds narrow-band noise of different frequencies |
