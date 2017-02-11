#ifndef DATE_H
#define DATE_H


class ParticleFilter
{
public:
    // Data Members
    int world_size;
    float x;
    float y;
    float orientation;
    float forward_noise;
    float turn_noise;
    float sense_noise;

    //Member Functions
    ParticleFilter();
    void set(float x, float y, float orientation);
    void set_noise(float forward_noise, float turn_noise, float sense_noise);
    std::vector<float> sense();
    void move(ParticleFilter pf,float turn, float forward);
    float Gaussian(float x, float mean, float variance);
    float measurement_prob();
    float eval();
};

#endif
