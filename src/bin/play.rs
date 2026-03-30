use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, MouseButton, MouseEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::execute;
use crossterm::event::{EnableMouseCapture, DisableMouseCapture};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Paragraph};

use hexo_rs::game::{GameConfig, MoveError};
use hexo_rs::{Coord, GameState, Player};

/// Layout constants for hex rendering.
/// Each hex cell is rendered as a 4-char wide, 2-row tall diamond:
///
/// ```text
///  /P1\
///  \__/
/// ```
///
/// Axial→screen mapping:
///   screen_x = center_x + q * CELL_W + r * (CELL_W / 2)
///   screen_y = center_y + r * ROW_H
const CELL_W: i32 = 4;
const ROW_H: i32 = 2;

struct App {
    game: GameState,
    cursor: Option<Coord>,
    message: String,
    config: GameConfig,
}

impl App {
    fn new(config: GameConfig) -> Self {
        Self {
            game: GameState::with_config(config),
            cursor: None,
            message: "Click a hex to place a stone. Press 'q' to quit, 'r' to restart.".into(),
            config,
        }
    }

    fn restart(&mut self) {
        self.game = GameState::with_config(self.config);
        self.cursor = None;
        self.message = "New game started.".into();
    }

    /// Convert screen (col, row) to axial (q, r) given the center of the viewport.
    fn screen_to_axial(&self, col: u16, row: u16, center_x: i32, center_y: i32) -> Coord {
        // Invert the mapping:
        //   r = (screen_y - center_y) / ROW_H
        //   q = (screen_x - center_x - r * CELL_W/2) / CELL_W
        let sy = row as i32 - center_y;
        let sx = col as i32 - center_x;

        // Round to nearest hex
        let r_f = sy as f64 / ROW_H as f64;
        let r = r_f.round() as i32;
        let q_f = (sx as f64 - r as f64 * (CELL_W as f64 / 2.0)) / CELL_W as f64;
        let q = q_f.round() as i32;
        (q, r)
    }

    fn try_place(&mut self, coord: Coord) {
        if self.game.is_terminal() {
            self.message = "Game is over! Press 'r' to restart.".into();
            return;
        }

        match self.game.apply_move(coord) {
            Ok(()) => {
                let stones = self.game.placed_stones().len();
                if self.game.is_terminal() {
                    match self.game.winner() {
                        Some(p) => self.message = format!("{p} wins! ({stones} stones). Press 'r' to restart."),
                        None => self.message = format!("Draw after {stones} stones. Press 'r' to restart."),
                    }
                } else {
                    let player = self.game.current_player().unwrap();
                    let remaining = self.game.moves_remaining_this_turn();
                    self.message = format!(
                        "{player} to move ({remaining} remaining). Placed at ({}, {}). {stones} stones.",
                        coord.0, coord.1
                    );
                }
            }
            Err(MoveError::CellOccupied) => {
                self.message = format!("({}, {}) is occupied!", coord.0, coord.1);
            }
            Err(MoveError::OutOfRange) => {
                self.message = format!("({}, {}) is out of range!", coord.0, coord.1);
            }
            Err(MoveError::GameOver) => {
                self.message = "Game is over! Press 'r' to restart.".into();
            }
        }
    }
}

fn render(frame: &mut Frame, app: &App) {
    let area = frame.area();

    // Reserve bottom 3 rows for status
    let [board_area, status_area] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(3),
    ])
    .areas(area);

    // Center of the board area
    let cx = board_area.x as i32 + board_area.width as i32 / 2;
    let cy = board_area.y as i32 + board_area.height as i32 / 2;

    // Draw all placed stones
    let stones = app.game.placed_stones();
    for &((q, r), player) in &stones {
        let sx = cx + q * CELL_W + r * (CELL_W / 2);
        let sy = cy + r * ROW_H;

        if sx >= board_area.x as i32
            && sx < (board_area.x + board_area.width) as i32 - 3
            && sy >= board_area.y as i32
            && sy + 1 < (board_area.y + board_area.height) as i32
        {
            let (label, color) = match player {
                Player::P1 => ("P1", Color::Blue),
                Player::P2 => ("P2", Color::Red),
            };

            // Top row: /XX\
            let top = format!("/{label}\\");
            frame.render_widget(
                Span::styled(&top, Style::default().fg(color).bold()),
                Rect::new(sx as u16, sy as u16, 4, 1),
            );
            // Bottom row: \__/
            frame.render_widget(
                Span::styled("\\__/", Style::default().fg(color)),
                Rect::new(sx as u16, (sy + 1) as u16, 4, 1),
            );
        }
    }

    // Draw legal move markers (dots)
    if !app.game.is_terminal() {
        let legal = app.game.legal_moves_set();
        for &(q, r) in legal.iter() {
            let sx = cx + q * CELL_W + r * (CELL_W / 2);
            let sy = cy + r * ROW_H;

            if sx >= board_area.x as i32
                && sx < (board_area.x + board_area.width) as i32 - 3
                && sy >= board_area.y as i32
                && sy + 1 < (board_area.y + board_area.height) as i32
            {
                // Only show nearby legal moves to avoid clutter
                let near_stone = stones.iter().any(|&((sq, sr), _)| {
                    let dq = (q - sq).abs();
                    let dr = (r - sr).abs();
                    dq.max(dr).max((dq + dr).abs()) <= 2
                });
                if near_stone {
                    let style = if app.cursor == Some((q, r)) {
                        Style::default().fg(Color::Yellow).bold()
                    } else {
                        Style::default().fg(Color::DarkGray)
                    };
                    frame.render_widget(
                        Span::styled(" .. ", style),
                        Rect::new(sx as u16, sy as u16, 4, 1),
                    );
                    frame.render_widget(
                        Span::styled(" .. ", style),
                        Rect::new(sx as u16, (sy + 1) as u16, 4, 1),
                    );
                }
            }
        }
    }

    // Highlight cursor
    if let Some((q, r)) = app.cursor {
        let sx = cx + q * CELL_W + r * (CELL_W / 2);
        let sy = cy + r * ROW_H;
        if sx >= board_area.x as i32
            && sx < (board_area.x + board_area.width) as i32 - 3
            && sy >= board_area.y as i32
            && sy + 1 < (board_area.y + board_area.height) as i32
        {
            frame.render_widget(
                Span::styled("[  ]", Style::default().fg(Color::Yellow).bold()),
                Rect::new(sx as u16, sy as u16, 4, 1),
            );
            frame.render_widget(
                Span::styled("[  ]", Style::default().fg(Color::Yellow).bold()),
                Rect::new(sx as u16, (sy + 1) as u16, 4, 1),
            );
        }
    }

    // Status bar
    let player_info = if let Some(p) = app.game.current_player() {
        let remaining = app.game.moves_remaining_this_turn();
        format!(" {p} to move | {remaining} moves left | {} stones ", app.game.placed_stones().len())
    } else {
        match app.game.winner() {
            Some(p) => format!(" {p} wins! "),
            None => " Draw! ".into(),
        }
    };

    let status = Paragraph::new(vec![
        Line::from(app.message.as_str()),
        Line::from(player_info),
        Line::from(" q: quit | r: restart | click: place stone | mouse hover: highlight "),
    ])
    .block(Block::bordered().title(" HeXO "));

    frame.render_widget(status, status_area);
}

fn main() -> io::Result<()> {
    // Parse optional CLI args for curriculum variants
    let args: Vec<String> = std::env::args().collect();
    let config = if args.len() > 1 {
        let win_len: u8 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6);
        let radius: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
        let max_moves: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
        GameConfig { win_length: win_len, placement_radius: radius, max_moves }
    } else {
        GameConfig::FULL_HEXO
    };

    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
    let mut terminal = ratatui::init();

    let mut app = App::new(config);

    loop {
        terminal.draw(|frame| render(frame, &app))?;

        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('r') => app.restart(),
                    _ => {}
                },
                Event::Mouse(mouse) => {
                    let area = terminal.get_frame().area();
                    let cx = area.x as i32 + area.width as i32 / 2;
                    let cy = area.y as i32 + (area.height as i32 - 3) / 2;

                    match mouse.kind {
                        MouseEventKind::Moved | MouseEventKind::Drag(MouseButton::Left) => {
                            let coord = app.screen_to_axial(mouse.column, mouse.row, cx, cy);
                            app.cursor = Some(coord);
                        }
                        MouseEventKind::Down(MouseButton::Left) => {
                            let coord = app.screen_to_axial(mouse.column, mouse.row, cx, cy);
                            app.try_place(coord);
                            app.cursor = Some(coord);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    ratatui::restore();
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;
    disable_raw_mode()?;
    Ok(())
}
