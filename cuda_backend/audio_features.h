#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AudioFeatureResult {
    int num_frames;
    int frame_size;
    int hop_size;
    int waveform_buckets;
    int num_samples;
    int sample_rate;
    float duration_seconds;

    float* rms;
    float* peak;
    float* bass;
    float* mid;
    float* treble;
    float* centroid;
    float* waveform_min;
    float* waveform_max;

    float h2d_ms;
    float rms_peak_ms;
    float window_pack_ms;
    float fft_ms;
    float band_energy_ms;
    float waveform_minmax_ms;
    float d2h_ms;
    float total_gpu_ms;
    float total_wall_ms;

    char error[512];
} AudioFeatureResult;

int compute_audio_features_cuda(
    const float* audio,
    int num_samples,
    int sample_rate,
    int frame_size,
    int hop_size,
    int waveform_buckets,
    AudioFeatureResult* out_result
);

void free_audio_feature_result(AudioFeatureResult* result);

#ifdef __cplusplus
}
#endif
