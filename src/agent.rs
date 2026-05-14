//! The part of the simulation which handles the ants that move around the grid and leave pheromones.

use std::{
    cmp::Ordering,
    f32::consts::{PI, TAU},
};

use crate::{
    config::{AgentConfig, Config, WallBounceReaction},
    grid::Simulation,
    random::{rand_f32, rand_symmetric_f32, rand_u32},
};

/// Current runtime state of an ant that moves around the grid and leaves pheromones.
pub(crate) struct Agent {
    /// The x-coordinate of the agent in pixels.
    pub(crate) x: f32,

    /// The y-coordinate of the agent in pixels.
    pub(crate) y: f32,

    /// The direction of the agent in radians.
    ///
    /// 0.0 faces towards the positive x-axis. Rotation is clockwise.
    direction: f32,

    /// The speed of the agent in pixels per update.
    speed: f32,

    /// The pheromone value of the agent.
    pub(crate) value: f32,

    /// current seed for the random number generator.
    rng: u32,

    /// The width of the sensor cone in radians.
    sensor_width: f32,

    /// The distance of the sensor cone in pixels.
    sensor_distance: f32,

    /// Factor by which the speed of anti-ants will be reduced compared to normal ants.
    anti_speed_factor: f32,

    /// Whether to flip the sign of the pheromone value when the agent hits a wall.
    wall_bounce_flip_value: bool,

    /// What to do, if the agent hits a wall.
    wall_bounce_reaction: WallBounceReaction,
}

impl Agent {
    /// Create a new agent with the given configuration.
    pub(crate) fn new(config: &Config, width: usize, height: usize, rng: &mut u32) -> Self {
        #![expect(
            clippy::suboptimal_flops,
            reason = "this would impair readability for code that is not performance critical"
        )]

        let AgentConfig {
            count: _,
            ref value,
            ref speed,
            sensor_width,
            sensor_distance,
            anti_percentage,
            anti_speed_factor,
            wall_bounce_flip_value,
            wall_bounce_reaction,
        } = config.agent;

        // compute an individual seed
        let mut rng = rand_u32(rng) ^ rand_u32(rng);
        let mut random = || rand_f32(&mut rng);

        #[expect(clippy::cast_precision_loss, reason = "dimensions are small enough")]
        let (width, height) = (width as f32, height as f32);

        let x = random() * width;
        let y = random() * height;
        let center_distance = f32::hypot(x - width * 0.5, y - height * 0.5);
        let radius = height * 0.1 / center_distance;
        let x = (x - width * 0.5) * radius + width * 0.5;
        let y = (y - height * 0.5) * radius + height * 0.5;
        // let (sin, cos) = (direction + PI).sin_cos();
        // let x = width as f32 * 0.5 + cos * r;
        // let y = height as f32 * 0.5 + sin * r;

        // let x = width as f32 * 0.5;
        // let y = height as f32 * 0.5;

        // let direction = random() * TAU;
        let direction = f32::atan2(y - height * 0.5, x - width * 0.5) - PI * 1.0;
        let direction = if direction.is_nan() { 0.0 } else { direction };
        // let direction = PI / 2.0;

        let speed = speed.start + random() * (speed.end - speed.start);
        let value = value.start + random() * (value.end - value.start);
        let sign = if random() > anti_percentage {
            1.0
        } else {
            -1.0
        };

        Self {
            x,
            y,
            direction,
            speed,
            value: value * sign,
            rng,
            sensor_width,
            sensor_distance,
            anti_speed_factor,
            wall_bounce_flip_value,
            wall_bounce_reaction,
        }
    }

    /// Update the agent's position and direction based on the current simulation state.
    pub(crate) fn update(&mut self, simulation: &Simulation) {
        self.update_direction(simulation);
        self.update_position(simulation);
    }

    /// Update the position of the agent based on its current direction and speed.
    ///
    /// This method will also handle wall collisions and the wall bounce reaction which might affect
    /// the direction and value.
    fn update_position(&mut self, simulation: &Simulation) {
        // make anti-ants slower than normal ants
        let scale = if self.value > 0.0 {
            1.0
        } else {
            self.anti_speed_factor
        };

        // move ant into the direction it is facing
        let (sin, cos) = self.direction.sin_cos();
        let new_x = cos.mul_add(self.speed * scale, self.x);
        let new_y = sin.mul_add(self.speed * scale, self.y);

        #[expect(clippy::cast_precision_loss, reason = "dimensions are small enough")]
        let (width, height) = (simulation.width as f32, simulation.height as f32);

        let x_rating = if new_x < 0.0 {
            Ordering::Less
        } else if new_x > width {
            Ordering::Greater
        } else {
            Ordering::Equal
        };

        let y_rating = if new_y < 0.0 {
            Ordering::Less
        } else if new_y > height {
            Ordering::Greater
        } else {
            Ordering::Equal
        };

        let hit_wall = !matches!((x_rating, y_rating), (Ordering::Equal, Ordering::Equal));

        if hit_wall {
            if self.wall_bounce_flip_value {
                self.value = -self.value;
            }

            match self.wall_bounce_reaction {
                WallBounceReaction::Center => (self.x, self.y) = (width / 2.0, height / 2.0),
                WallBounceReaction::Random => {
                    self.x = rand_f32(&mut self.rng) * width;
                    self.y = rand_f32(&mut self.rng) * height;
                }
                WallBounceReaction::WrapAround => {
                    self.x = match x_rating {
                        Ordering::Less => new_x + width,
                        Ordering::Greater => new_x - width,
                        Ordering::Equal => new_x,
                    };
                    self.y = match y_rating {
                        Ordering::Less => new_y + height,
                        Ordering::Greater => new_y - height,
                        Ordering::Equal => new_y,
                    };
                }
                WallBounceReaction::Clip => {
                    self.x = match x_rating {
                        Ordering::Less => 0.0,
                        Ordering::Greater => width,
                        Ordering::Equal => new_x,
                    };
                    self.y = match y_rating {
                        Ordering::Less => 0.0,
                        Ordering::Greater => height,
                        Ordering::Equal => new_y,
                    };
                }
                WallBounceReaction::FaceAway(spread) => {
                    let mut new_direction = |normal: f32| {
                        normal.mul_add(TAU, rand_symmetric_f32(&mut self.rng) * spread)
                    };

                    match x_rating {
                        Ordering::Less => {
                            self.direction = new_direction(0.0);
                            self.x = 0.0;
                        }
                        Ordering::Greater => {
                            self.direction = new_direction(0.5);
                            self.x = width;
                        }
                        Ordering::Equal => {
                            self.x = new_x;
                        }
                    }
                    match y_rating {
                        Ordering::Less => {
                            self.direction = new_direction(0.25);
                            self.y = 0.0;
                        }
                        Ordering::Greater => {
                            self.direction = new_direction(0.75);
                            self.y = height;
                        }
                        Ordering::Equal => {
                            self.y = new_y;
                        }
                    }
                }
                WallBounceReaction::BounceOff => {
                    match x_rating {
                        Ordering::Less => {
                            self.direction = 0.25_f32.mul_add(TAU, -self.direction);
                            self.x = 0.0;
                        }
                        Ordering::Greater => {
                            self.direction = 0.75_f32.mul_add(TAU, -self.direction);
                            self.x = width;
                        }
                        Ordering::Equal => {
                            self.x = new_x;
                        }
                    }
                    match y_rating {
                        Ordering::Less => {
                            self.direction = 0.50_f32.mul_add(TAU, -self.direction);
                            self.y = 0.0;
                        }
                        Ordering::Greater => {
                            self.direction = 0.0_f32.mul_add(TAU, -self.direction);
                            self.y = height;
                        }
                        Ordering::Equal => {
                            self.y = new_y;
                        }
                    }
                }
            }
        } else {
            self.x = new_x;
            self.y = new_y;
        }
    }

    /// Update the direction (orientation) of the agent based on pheromone levels around it.
    fn update_direction(&mut self, simulation: &Simulation) {
        let sniff = |angle: f32| {
            let (sin, cos) = angle.sin_cos();
            simulation
                .read_buffer
                .cell(
                    self.sensor_distance.mul_add(cos, self.x),
                    self.sensor_distance.mul_add(sin, self.y),
                )
                .level
        };

        #[expect(clippy::neg_multiply, reason = "improves readability")]
        let sniffs = [
            self.sensor_width * -1.0,
            self.sensor_width * -0.5,
            self.sensor_width * 0.5,
            self.sensor_width * 1.0,
        ];

        let mut angle_sum = 0.0;
        for angle in sniffs {
            angle_sum += sniff(self.direction + angle) * angle;
        }

        let delta = angle_sum;
        let jitter = rand_symmetric_f32(&mut self.rng) * self.sensor_width;
        self.direction += delta * 0.5 + jitter * 0.3;
    }
}
