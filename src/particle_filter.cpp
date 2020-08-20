/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

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
using std::normal_distribution;
using std::uniform_real_distribution;
std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  
  //create normal distributions for x,y,theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  //Create particles in for loop
  for (int i=0; i < num_particles; ++i){
	  Particle myparticle;
	  myparticle.id = i;
	  myparticle.x = dist_x(gen); //sample from gaussian with generator
	  myparticle.y = dist_y(gen);
	  myparticle.theta = dist_theta(gen);
	  myparticle.weight = 1;

	  particles.push_back(myparticle); //Append each particle too the set of current particles
  }
  //Set is initalized to true
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  //create normal distributions for measurements
  normal_distribution<double> ND_x(0, std_pos[0]);
  normal_distribution<double> ND_y(0, std_pos[1]);
  normal_distribution<double> ND_theta(0, std_pos[2]);

	for (int i = 0; i < particles.size(); i++){
		//when yaw rate is zero
		if(fabs(yaw_rate) < 0.001){
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		} else { //when the yaw rate is not zero
			particles[i].x += velocity / yaw_rate * ( sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta) );
			particles[i].y += velocity / yaw_rate * ( cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		//Add noise
		particles[i].x += ND_x(gen);
		particles[i].y += ND_y(gen);
		particles[i].theta += ND_theta(gen);	
	}
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
   
   //Loop through observations
   for (int i=0; i<observations.size(); i++){

	   double shortest_dist = std::numeric_limits<double>::max(); //set inital shortest distance to infinite
	   int id = -1; //Initial id of shortest distance measurement

	   //For each observation loop through the predictions
	   for (int j=0; j<predicted.size(); j++){
		   double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
		   if (distance < shortest_dist){
			   shortest_dist = distance;
			   id = predicted[j].id;
		   }
		
		observations[i].id = id;
	   }
   }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
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
		
	//For each particle
	for (int i=0; i<num_particles; i++){
		double x_p = particles[i].x;
		double y_p = particles[i].y;
		double theta_p = particles[i].theta;
		
		vector<LandmarkObs> predicted;	
		vector<LandmarkObs> transformed_obs;	
		
		//Go through the list of landmarks
		for (int j=0; j<map_landmarks.landmark_list.size(); j++){
			
			//Get the data from the j-th landmark
			double x_lm = map_landmarks.landmark_list[j].x_f;
			double y_lm = map_landmarks.landmark_list[j].y_f;
			double id_lm = map_landmarks.landmark_list[j].id_i;
			//If the landmark is not within the sensor range from the particle ...
			//... it can be skipped. If it is within the range, it is added to the predicted variable
			if (dist(x_p,y_p, x_lm, y_lm) <= sensor_range) {
				LandmarkObs valid_lm;
				valid_lm.x = x_lm;
				valid_lm.y = y_lm;
				valid_lm.id = id_lm;
				predicted.push_back(valid_lm);
			}
		}
		
		//Transform each observation into map coordinates
		for (int k=0; k<observations.size(); k++){
			LandmarkObs trans;
			trans.x = cos(theta_p) * observations[k].x - sin(theta_p) * observations[k].y + x_p;
			trans.y = sin(theta_p) * observations[k].x + cos(theta_p) * observations[k].y + y_p;
			trans.id = observations[k].id;
			transformed_obs.push_back(trans);
		}
		
		//Landmark Association
		dataAssociation(predicted, transformed_obs);
		particles[i].weight = 1;
		
		//Weight update
		for (int ii=0; ii<transformed_obs.size(); ii++){
			double landmarkX, landmarkY;
			double obs_x = transformed_obs[ii].x;
			double obs_y = transformed_obs[ii].y;
			int obs_id = transformed_obs[ii].id;
			
			for (int jj=0; jj<predicted.size(); jj++){
				if (predicted[jj].id == obs_id){
					landmarkX = predicted[jj].x;
					landmarkY = predicted[jj].y;
				}
			}
		
		double A = (1/(2*M_PI*std_landmark[0]*std_landmark[1]));
		double B = exp( -( pow((obs_x - landmarkX),2)/(2*pow(std_landmark[0],2)) + pow((obs_y - landmarkY),2)/(2*pow(std_landmark[1],2))));
		double new_weight = A * B;
		
		particles[i].weight *= new_weight;
		}
	}
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
	
	vector<double> weights;
	vector<Particle> new_particles;
	double beta = 0;
	double maxweight = 0;
	
	//Determine maxweight
	for (int i=0; i<num_particles; i++){
		weights.push_back(particles[i].weight);
		if(particles[i].weight > maxweight){
			maxweight=particles[i].weight;
		}
	}
	
	//Create distributions for resampling wheel
	uniform_real_distribution<float> ND_w(0.0, maxweight);
	uniform_real_distribution<float> ND_p(0, num_particles-1);
	int index = ND_p(gen);
	
	for (int j=0; j < num_particles; j++){
		beta += ND_w(gen)*2;
		while (beta > weights[index]){
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}
	particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}