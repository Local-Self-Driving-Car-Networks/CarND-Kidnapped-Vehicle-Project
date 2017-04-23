/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang & John Lees-Miller
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

// 50 is not that many particles, but it seems to be enough to provide good
// tracking on the project dataset.
const size_t NUM_PARTICLES = 50;

// This would probably live more naturally as a class member, but since we
// can only submit this file (and therefore can't change the header file
// substantially), let's just use a static generator.
static std::default_random_engine generator(42);

// Print out a particle for tracing.
std::ostream &operator<<(std::ostream& os, const Particle &particle) {
  return os << particle.x << ' ' << particle.y << ' ' << particle.theta <<
    ' ' << particle.weight;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position
	// (based on estimates of x, y, theta and their uncertainties from GPS) and
	// all weights to 1. Add random Gaussian noise to each particle.
  std::normal_distribution<double> x_normal(x, std[0]);
  std::normal_distribution<double> y_normal(y, std[1]);
  std::normal_distribution<double> theta_normal(theta, std[2]);

	for (size_t i = 0; i < NUM_PARTICLES; ++i) {
		Particle particle;
		particle.id = -1;
		particle.x = x_normal(generator);
		particle.y = y_normal(generator);
		particle.theta = theta_normal(generator);
		particle.weight = 1;
		particles.push_back(particle);
	}

  num_particles = NUM_PARTICLES;
  is_initialized = true;
}

const double EPSILON = 1e-6;

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  std::normal_distribution<double> x_normal(0, std_pos[0]);
  std::normal_distribution<double> y_normal(0, std_pos[1]);
  std::normal_distribution<double> theta_normal(0, std_pos[2]);

	// Add measurements to each particle and add random Gaussian noise.
	for (std::vector<Particle>::iterator particle = particles.begin();
		particle != particles.end(); ++particle)
	{
		if (fabs(yaw_rate) < EPSILON) {
			particle->x += velocity * delta_t * cos(particle->theta);
			particle->y += velocity * delta_t * sin(particle->theta);
		} else {
			double r = velocity / yaw_rate;
			double new_theta = particle->theta + yaw_rate * delta_t;
			particle->x += r * (sin(new_theta) - sin(particle->theta));
			particle->y += r * (cos(particle->theta) - cos(new_theta));
			particle->theta = new_theta;
		}
		particle->x += x_normal(generator);
		particle->y += y_normal(generator);
		particle->theta += theta_normal(generator);
	}
}

void ParticleFilter::dataAssociation(
  std::vector<LandmarkObs> predicted,
  std::vector<LandmarkObs>& observations)
{
	// Find the predicted measurement that is closest to each observed measurement
	// and assign the observed measurement to this particular landmark.
  for (size_t i = 0; i < observations.size(); ++i) {
    size_t closest_j = std::numeric_limits<size_t>::max();
    double closest_distance = std::numeric_limits<double>::max();
    for (size_t j = 0; j < predicted.size(); ++j) {
      double distance_j = dist(
        observations[i].x, observations[i].y,
        predicted[j].x, predicted[j].y
      );
      if (distance_j < closest_distance) {
        closest_distance = distance_j;
        closest_j = j;
      }
    }
    observations[i].id = closest_j;
  }
}

/**
 * Calculate the log of the multivariate normal probability density function.
 * Using the logarithm should help us avoid underflow errors for very small
 * probabilities.
 *
 * @param  x0 coordinate to evaluate at
 * @param  x1 coordinate to evaluate at
 * @param  mean0 distribution mean
 * @param  mean1 distribution mean
 * @param  stdev0 distribution standard deviation
 * @param  stdev1 distribution standard deviation
 * @return log probability
 */
double mvnormal_log_density(double x0, double x1,
  double mean0, double mean1, double stdev0, double stdev1)
{
  double scale = log(2 * stdev0 * stdev1 * M_PI);
  double d0 = (x0 - mean0) * (x0 - mean0) / (stdev0 * stdev0);
  double d1 = (x1 - mean1) * (x1 - mean1) / (stdev1 * stdev1);
  return -0.5 * (scale + d0 + d1);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks)
{
  std::vector<LandmarkObs> transformed_observations(observations.size());

  // Get map landmarks into observation format so we can pass them to the
  // dataAssociation function. It seems odd that we have to do this, but we
  // can't change the header file.
  std::vector<LandmarkObs> map_observations(map_landmarks.landmark_list.size());
  for (size_t i = 0; i < map_observations.size(); ++i) {
    map_observations[i].id = 0; // not used
    map_observations[i].x = map_landmarks.landmark_list[i].x_f;
    map_observations[i].y = map_landmarks.landmark_list[i].y_f;
  }

	for (std::vector<Particle>::iterator particle = particles.begin();
		particle != particles.end(); ++particle)
	{
    // Transform observations from vehicle coordinates to global map
    // coordinates.
    for (size_t i = 0; i < observations.size(); ++i) {
      transformed_observations[i].x =
        observations[i].x * cos(particle->theta) -
        observations[i].y * sin(particle->theta) +
        particle->x;
      transformed_observations[i].y =
        observations[i].x * sin(particle->theta) +
        observations[i].y * cos(particle->theta) +
        particle->y;
    }

    // Find nearest neighbors.
    dataAssociation(map_observations, transformed_observations);

    // Find weight for each particle. Calculate in log space to avoid numerical
    // underflow issues, and then exponentiate to get a probability out at the
    // end.
    double p = 0.0;
    for (size_t i = 0; i < observations.size(); ++i) {
      int j = transformed_observations[i].id;
      p += mvnormal_log_density(
        transformed_observations[i].x, transformed_observations[i].y,
        map_observations[j].x, map_observations[j].y,
        std_landmark[0], std_landmark[1]);
    }
    particle->weight = exp(p);

    // Uncomment these two `cerr` lines to enable tracing on stderr. This slows
    // the program down a lot, so I have turned it off for submission.
    // std::cerr << *particle << ' ';
  }
  // std::cerr << std::endl;
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their
	// weight. Use a wheel sampler.
  double max_weight = 0;
	for (std::vector<Particle>::const_iterator particle = particles.begin();
		particle != particles.end(); ++particle)
  {
    if (particle->weight > max_weight) {
      max_weight = particle->weight;
    }
  }

  std::uniform_int_distribution<size_t> uniform_start(0, particles.size() - 1);
  std::uniform_real_distribution<double> uniform_spin(0, 2 * max_weight);

  size_t index = uniform_start(generator);
  double beta = 0;
  std::vector<Particle> new_particles;
  new_particles.reserve(particles.size());
  for (size_t i = 0; i < particles.size(); ++i) {
    beta += uniform_spin(generator);
    while (particles[index].weight < beta) {
      beta -= particles[index].weight;
      index = (index + 1) % particles.size();
    }

    Particle new_particle;
    new_particle.id = -1;
    new_particle.x = particles[index].x;
    new_particle.y = particles[index].y;
    new_particle.theta = particles[index].theta;
    new_particle.weight = 1;
    new_particles.push_back(new_particle);
  }

  particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
