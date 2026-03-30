use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, MouseButton, MouseEventKind};
use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Paragraph};

use hexo_rs::game::{GameConfig, MoveError};
use hexo_rs::{Coord, GameState, Player};

/// Each hex cell is 4 chars wide, 2 rows tall:
///   / X\      / O\      /  \
///   \__/      \__/      \__/
///
/// Axial→screen:
///   sx = center_x + q * CELL_W + r * (CELL_W / 2)
///   sy = center_y + r * ROW_H
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

    fn screen_to_axial(&self, col: u16, row: u16, center_x: i32, center_y: i32) -> Coord {
        let sy = row as i32 - center_y;
        let sx = col as i32 - center_x;
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
                        Some(p) => {
                            let sym = match p {
                                Player::P1 => "X",
                                Player::P2 => "O",
                            };
                            self.message =
                                format!("{sym} wins! ({stones} stones). Press 'r' to restart.");
                        }
                        None => {
                            self.message =
                                format!("Draw after {stones} stones. Press 'r' to restart.");
                        }
                    }
                } else {
                    let sym = match self.game.current_player().unwrap() {
                        Player::P1 => "X",
                        Player::P2 => "O",
                    };
                    let remaining = self.game.moves_remaining_this_turn();
                    self.message = format!(
                        "{sym} to move ({remaining} remaining). Placed at ({}, {}). {stones} stones.",
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

fn in_board(sx: i32, sy: i32, area: Rect) -> bool {
    sx >= area.x as i32
        && sx + 3 < (area.x + area.width) as i32
        && sy >= area.y as i32
        && sy + 1 < (area.y + area.height) as i32
}

fn draw_hex(frame: &mut Frame, sx: i32, sy: i32, top: &str, bot: &str, style: Style) {
    frame.render_widget(
        Span::styled(top, style),
        Rect::new(sx as u16, sy as u16, 4, 1),
    );
    frame.render_widget(
        Span::styled(bot, style),
        Rect::new(sx as u16, (sy + 1) as u16, 4, 1),
    );
}

fn render(frame: &mut Frame, app: &App) {
    let area = frame.area();

    let [board_area, status_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(3)]).areas(area);

    let cx = board_area.x as i32 + board_area.width as i32 / 2;
    let cy = board_area.y as i32 + board_area.height as i32 / 2;

    let stones = app.game.placed_stones();
    let legal = app.game.legal_moves_set();

    // 1. Draw empty hex outlines for nearby legal moves
    if !app.game.is_terminal() {
        for &(q, r) in legal.iter() {
            let sx = cx + q * CELL_W + r * (CELL_W / 2);
            let sy = cy + r * ROW_H;
            if !in_board(sx, sy, board_area) {
                continue;
            }

            // Only show hexes within distance 2 of any stone to avoid clutter
            let near = stones
                .iter()
                .any(|&((sq, sr), _)| hexo_rs::hex::hex_distance((q, r), (sq, sr)) <= 2);
            if !near {
                continue;
            }

            let style = Style::default().fg(Color::DarkGray);
            draw_hex(frame, sx, sy, "/  \\", "\\__/", style);
        }
    }

    // 2. Draw placed stones (on top of empty hexes)
    for &((q, r), player) in &stones {
        let sx = cx + q * CELL_W + r * (CELL_W / 2);
        let sy = cy + r * ROW_H;
        if !in_board(sx, sy, board_area) {
            continue;
        }

        let (top, color) = match player {
            Player::P1 => ("/ X\\", Color::Cyan),
            Player::P2 => ("/ O\\", Color::Magenta),
        };
        draw_hex(frame, sx, sy, top, "\\__/", Style::default().fg(color).bold());
    }

    // 3. Draw cursor highlight (on top of everything)
    if let Some((q, r)) = app.cursor {
        let sx = cx + q * CELL_W + r * (CELL_W / 2);
        let sy = cy + r * ROW_H;
        if in_board(sx, sy, board_area) {
            let style = Style::default().fg(Color::Yellow).bold();
            // Show cursor as a highlighted hex outline, preserving what's inside
            let is_stone = stones.iter().any(|&(c, _)| c == (q, r));
            let is_legal = legal.contains(&(q, r));

            if is_stone {
                // Don't overwrite stones, just add corner highlights
                frame.render_widget(
                    Span::styled(">", style),
                    Rect::new(sx as u16, sy as u16, 1, 1),
                );
                frame.render_widget(
                    Span::styled("<", style),
                    Rect::new((sx + 3) as u16, sy as u16, 1, 1),
                );
            } else if is_legal {
                draw_hex(frame, sx, sy, "/\u{b7}\u{b7}\\", "\\__/", style);
            } else {
                draw_hex(frame, sx, sy, "/  \\", "\\__/", style);
            }
        }
    }

    // Status bar
    let player_info = if let Some(p) = app.game.current_player() {
        let sym = match p {
            Player::P1 => "X",
            Player::P2 => "O",
        };
        let remaining = app.game.moves_remaining_this_turn();
        format!(
            " {sym} to move | {remaining} moves left | {} stones ",
            app.game.placed_stones().len()
        )
    } else {
        match app.game.winner() {
            Some(Player::P1) => " X wins! ".into(),
            Some(Player::P2) => " O wins! ".into(),
            None => " Draw! ".into(),
        }
    };

    let status = Paragraph::new(vec![
        Line::from(app.message.as_str()),
        Line::from(player_info),
        Line::from(" q: quit | r: restart | click: place stone "),
    ])
    .block(Block::bordered().title(" HeXO "));

    frame.render_widget(status, status_area);
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let config = if args.len() > 1 {
        let win_len: u8 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6);
        let radius: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
        let max_moves: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
        GameConfig {
            win_length: win_len,
            placement_radius: radius,
            max_moves,
        }
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
