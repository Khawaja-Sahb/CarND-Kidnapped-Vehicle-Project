
  
#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
std::default_random_engine gen;
void ParticleFilter::init(double x, double y, double theta, double std[])
{
  	  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 20;  // TODO: Set the number of particles
  
  std::normal_distribution<double> distx(x,std[0]);
  std::normal_distribution<double> disty(y,std[1]);
  std::normal_distribution<double> disttheta(theta,std[2]);
  
  particles.resize(num_particles);
  weights.resize(num_particles);
  
  double init_weight=1.0/num_particles;
  
  for(int i=0;i<num_particles;i++){
    particles[i].id=i;
    particles[i].x=distx(gen);
    particles[i].y=disty(gen);
    particles[i].theta=disttheta(gen);
    particles[i].weight=init_weight;
  }
  
  is_initialized=true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
    /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::normal_distribution<double> distx(0.0,std_pos[0]);
  std::normal_distribution<double> disty(0.0,std_pos[1]);
  std::normal_distribution<double> disttheta(0.0,std_pos[2]);
  
  for(int i=0;i<num_particles;i++){
    if(fabs(yaw_rate)<0.001){
      particles[i].x+=velocity*delta_t*cos(particles[i].theta);
      particles[i].y+=velocity*delta_t*sin(particles[i].theta);
    }else{
      particles[i].x+=(velocity/yaw_rate)*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
      particles[i].y+=(velocity/yaw_rate)*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
      particles[i].theta+=yaw_rate*delta_t;
    }
    
    particles[i].x+=distx(gen);
    particles[i].y+=disty(gen);
    particles[i].theta+=disttheta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations)
{
   /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
    for(unsigned int i=0;i<observations.size();i++){
      LandmarkObs o=observations[i];
      double min_dist = std::numeric_limits<double>::max();
      int mapid=-1;
      for(unsigned int j=0;j<observations.size();j++){
        LandmarkObs p=predicted[j];
        double distance=dist(o.x,o.y,p.x,p.y);
        if(distance<min_dist){
          min_dist=distance;
          mapid=p.id;
        }
      }
      observations[i].id=mapid;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const vector<LandmarkObs>& observations,
                                   const Map& map_landmarks)
{
    /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

    for (int i = 0; i < num_particles; i++){
        // Assign individual components to matrix transform.
      double xp = particles[i].x;
      double yp = particles[i].y;
      double theta = particles[i].theta;
      vector<LandmarkObs> tf_observations;
      for(unsigned int j=0;j<observations.size();j++){
        LandmarkObs Ob;
        Ob.id=j;
        Ob.x=particles[i].x+cos(particles[i].theta)*observations[j].x-sin(particles[i].theta)*observations[j].y;
        Ob.y=particles[i].y+sin(particles[i].theta)*observations[j].x+cos(particles[i].theta)*observations[j].y;
        tf_observations.push_back(Ob);
      }

        // Filter landmarks to only that which is within the sensor range
      vector<LandmarkObs> landmarks_in_range;
      for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); k++){
        double distance = dist(xp, yp, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);
        if (distance <= sensor_range){
          LandmarkObs landmark_in_range;
          landmark_in_range.x = map_landmarks.landmark_list[k].x_f;
          landmark_in_range.y = map_landmarks.landmark_list[k].y_f;
          landmark_in_range.id = map_landmarks.landmark_list[k].id_i;
          landmarks_in_range.push_back(landmark_in_range);
        }
      }

      // Use dataAssociation to update the observations with the ID of the nearest landmark
      dataAssociation(landmarks_in_range, tf_observations);
      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];
      double var_x = sig_x * sig_x;
      double var_y = sig_y * sig_y;
      double gaussian_norm = 1.0 / (2.0 * M_PI * sig_x * sig_y);
      // Calculate the weight of each particle using multivariate gaussian probability density function
      particles[i].weight=1;
      double probability =1.0;
      for(unsigned int l=0;l<tf_observations.size();l++){
        double tx=tf_observations[l].x;
        double ty=tf_observations[l].y;
        double tid=tf_observations[l].id;
        for(unsigned int m=0;m<landmarks_in_range.size();m++){
          double px=landmarks_in_range[m].x;
          double py=landmarks_in_range[m].y;
          double pid=landmarks_in_range[m].id;
          if(tid==pid){
            double exponent_x = ((tx - px) * (tx -px)) / (2.0 * var_x);
            double exponent_y = ((ty - py) * (ty - py)) / (2.0 * var_y);
            probability = probability * gaussian_norm * exp(-(exponent_x + exponent_y));
            break;
          }
        }
      }
      particles[i].weight = probability;
      weights[i] = probability;
    }
    // Normalise the probability
    double weight_normalizer = std::accumulate(weights.begin(), weights.end(), 0.0f);
    for (int p = 0; p < num_particles; ++p){
      particles[p].weight = particles[p].weight / weight_normalizer;
      weights[p] = particles[p].weight;
    }
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  std::vector<Particle> new_particles;
  double weight_max=2**max_element(weights.begin(),weights.end());
  std::uniform_int_distribution<int>disun(0.0,num_particles-1);
  std::uniform_real_distribution<double>diswt(0.0,weight_max);
  auto index=disun(gen);
  double beta=0;
  for(unsigned int o=0;o<num_particles;o++){
    beta+=diswt(gen);
    while(beta>weights[index]){
      beta-=weights[index];
      index=(index+1)%num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  particles=new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, const vector<int>& associations, const vector<double>& sense_x,
                                     const vector<double>& sense_y)
{
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
    vector<double> v;

    if (coord == "X")
    {
        v = best.sense_x;
    }
    else
    {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
