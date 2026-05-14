//! The _backplane_ of the simulation which stores the pheromone levels for each cell in the grid.

use std::mem;

use rayon::{
    iter::{IndexedParallelIterator as _, ParallelIterator as _},
    slice::ParallelSliceMut as _,
};

use crate::{
    Agent,
    config::{Config, GridTopology, WorldConfig},
};

/// Contains the data for a single cell (pixel) in the grid.
#[derive(Default, Clone, Copy)]
pub(crate) struct Cell {
    /// The pheromone level of the cell.
    ///
    /// This level is typically between -1.0 and 1.0. Positive values will attract ants, negative
    /// values will repel them.
    pub(crate) level: f32,
}

/// Data structure holding the cell values for each pixel in the grid.
pub(crate) struct Grid {
    /// Width of the grid in cells/pixels.
    width: usize,

    /// Height of the grid in cells/pixels.
    height: usize,

    /// The actual cells/pixels in the grid.
    pub(crate) cells: Vec<Cell>,

    /// The topology of the grid.
    ///
    /// This determines how the edges of the grid are handled.
    topology: GridTopology,
}

impl Grid {
    /// Create a new grid with the given width and height in cells/pixels.
    fn new(width: usize, height: usize, topology: GridTopology) -> Self {
        Self {
            width,
            height,
            cells: vec![Cell::default(); width * height],
            topology,
        }
    }

    /// Map a row index to the actual row index in the grid.
    ///
    /// The topology will determine whether an out-of-bounds index shall be wrapped around or
    /// clamped to the edge.
    fn map_row(&self, y: i32) -> usize {
        let height = self.height as i32;
        match self.topology {
            GridTopology::Torus => {
                if (0..height).contains(&y) {
                    y as usize
                } else {
                    y.rem_euclid(height) as usize
                }
            }
            GridTopology::Plane => y.clamp(0, height - 1) as usize,
        }
    }

    /// Map a column index to the actual column index in the grid.
    ///
    /// The topology will determine whether an out-of-bounds index shall be wrapped around or
    /// clamped to the edge.
    fn map_col(&self, x: i32) -> usize {
        let width = self.width as i32;
        match self.topology {
            GridTopology::Torus => {
                if (0..width).contains(&x) {
                    x as usize
                } else {
                    x.rem_euclid(width) as usize
                }
            }
            GridTopology::Plane => x.clamp(0, width - 1) as usize,
        }
    }

    /// Get a row of cells from the grid.
    ///
    /// Out-of-bounds indices will be handled according to the topology.
    pub(crate) fn row(&self, y: i32) -> &[Cell] {
        #[expect(
            clippy::indexing_slicing,
            reason = "`map_row` ensures that the index is in bounds"
        )]
        &self.cells[self.map_row(y) * self.width..][..self.width]
    }

    /// Get a mutable row of cells from the grid.
    ///
    /// Out-of-bounds indices will be handled according to the topology.
    fn row_mut(&mut self, y: i32) -> &mut [Cell] {
        let mapped_row = self.map_row(y);
        #[expect(
            clippy::indexing_slicing,
            reason = "`map_row` ensures that the index is in bounds"
        )]
        &mut self.cells[mapped_row * self.width..][..self.width]
    }

    /// Get the cell index for the given x and y coordinates.
    ///
    /// Out-of-bounds indices will be handled according to the topology.
    ///
    /// The returned index is guaranteed to be within the bounds of the grid.
    fn index(&self, x: f32, y: f32) -> usize {
        // FIXME this should use the mapping methods as well
        let x = (x.round() as usize).clamp(0, self.width - 1);
        let y = (y.round() as usize).clamp(0, self.height - 1);
        y * self.width + x
    }

    /// Get the cell at the given x and y coordinates.
    ///
    /// Out-of-bounds indices will be handled according to the topology.
    // TODO implement interpolation
    pub(crate) fn cell(&self, x: f32, y: f32) -> &Cell {
        let index = self.index(x, y);
        #[expect(
            clippy::indexing_slicing,
            reason = "The `index` method ensures that the index is in bounds"
        )]
        &self.cells[index]
    }

    /// Get the mutable cell at the given x and y coordinates.
    ///
    /// Out-of-bounds indices will be handled according to the topology.
    // TODO implement interpolation
    fn cell_mut(&mut self, x: f32, y: f32) -> &mut Cell {
        let index = self.index(x, y);
        #[expect(
            clippy::indexing_slicing,
            reason = "The `index` method ensures that the index is in bounds"
        )]
        &mut self.cells[index]
    }

    /// Update the grid by blurring the pheromone levels of the read buffer.
    ///
    /// The decay factor will determine how much the pheromone levels will be reduced.
    fn blur(&mut self, read_buffer: &Self, decay_factor: f32) {
        self.cells
            .par_chunks_exact_mut(self.width)
            .enumerate()
            .for_each(|(y, write_row)| {
                // 5 rows around the current row
                let y = y as i32;
                let row = [
                    read_buffer.row(y - 1),
                    read_buffer.row(y),
                    read_buffer.row(y + 1),
                ];
                for (x, write_cell) in write_row.iter_mut().enumerate() {
                    // column indices for the 3 columns around x
                    let x = x as i32;
                    let col = [
                        read_buffer.map_col(x - 1),
                        read_buffer.map_col(x),
                        read_buffer.map_col(x + 1),
                    ];

                    #[expect(
                        clippy::indexing_slicing,
                        reason = "all indices are either compile-time constants or are guaranteed to be in bounds by using the mapping methods"
                    )]
                    let cell = |x_index: usize, y_index: usize| row[y_index][col[x_index]].level;

                    // filter kernel (weight sum = 16)
                    // 1 2 1
                    // 2 4 2
                    // 1 2 1

                    let value00 = cell(0, 0); // top left
                    let value01 = cell(1, 0); // top center
                    let value02 = cell(2, 0); // top right
                    let value10 = cell(0, 1); // left center
                    let value11 = cell(1, 1); // center
                    let value12 = cell(2, 1); // right center
                    let value20 = cell(0, 2); // bottom left
                    let value21 = cell(1, 2); // bottom center
                    let value22 = cell(2, 2); // bottom right

                    // sum up smallest values first for improved numerical stability
                    let corners = (value00 + value02 + value20 + value22) * 16.0_f32.recip();
                    let sides = (value01 + value10 + value21 + value12) * 8.0_f32.recip();
                    let center = value11 * 4.0_f32.recip();
                    let sum = corners + sides + center;
                    let level = sum * decay_factor;

                    // avoid sub-normal numbers for performance reasons
                    write_cell.level = if level.is_normal() { level } else { 0.0 };
                }
            });
    }
}

/// The current run-time state of the entire simulation.
pub(crate) struct Simulation {
    /// Width of the grid in cells/pixels.
    pub(crate) width: usize,

    /// Height of the grid in cells/pixels.
    pub(crate) height: usize,

    /// The buffer that is read from in the current frame.
    ///
    /// This will be swapped with the write buffer after each frame.
    pub(crate) read_buffer: Grid,

    /// The buffer that is written to in the current frame.
    ///
    /// This will be swapped with the read buffer after each frame.
    pub(crate) write_buffer: Grid,

    /// The value to use for the outermost pixel rows and columns.
    ///
    /// This will repel or attract the ants from the edges of the grid.
    /// A value of `None` means that the edges have no special effect.
    pub(crate) wall_value: Option<f32>,

    /// The decay factor to use for the pheromone levels.
    pub(crate) decay_factor: f32,
}

impl Simulation {
    /// Create a new simulation with the given width and height in cells/pixels.
    pub(crate) fn new(width: usize, height: usize, config: &Config) -> Self {
        let WorldConfig {
            wall_value,
            topology,
            decay_factor,
        } = config.world;

        Self {
            width,
            height,
            read_buffer: Grid::new(width, height, topology),
            write_buffer: Grid::new(width, height, topology),
            wall_value,
            decay_factor,
        }
    }

    /// Update the simulation by blurring the pheromone levels of the read buffer and writing them to the write buffer.
    pub(crate) fn blur(&mut self) {
        self.write_buffer.blur(&self.read_buffer, self.decay_factor);
    }

    /// Update the simulation by adding the pheromone levels of the agents to the write buffer.
    ///
    /// This will also apply the wall value to the outermost pixel rows and columns.
    pub(crate) fn update(&mut self, agents: &[Agent]) {
        for agent in agents {
            let level = &mut self.write_buffer.cell_mut(agent.x, agent.y).level;
            *level = (*level + agent.value).clamp(-1.0, 1.0);
        }

        // repulse from or attract to walls
        if let Some(value) = self.wall_value {
            // top wall
            self.write_buffer.row_mut(0).iter_mut().for_each(|cell| {
                cell.level = value;
            });

            // bottom wall
            self.write_buffer
                .row_mut(self.height as i32 - 1)
                .iter_mut()
                .for_each(|cell| {
                    cell.level = value;
                });

            // sides
            for y in 0..self.height {
                let row = self.write_buffer.row_mut(y as i32);
                if let Some(first) = row.first_mut() {
                    // left wall
                    first.level = value;
                }
                if let Some(last) = row.last_mut() {
                    // right wall
                    last.level = value;
                }
            }
        }

        // let row = (self.height / 2 - 1) * self.width;
        // for x in 0..self.width {
        //     // Hot coals + noise along the base.
        //     let bump = rand_u32(rng) % 96;
        //     self.cells[row + x].level = (220.0 - bump as f32) / 255.0;
        // }
    }

    /// Swap the read and write buffers.
    ///
    /// This will be called after each frame to prepare for the next frame.
    pub(crate) const fn swap_buffers(&mut self) {
        mem::swap(&mut self.read_buffer, &mut self.write_buffer);
    }
}
