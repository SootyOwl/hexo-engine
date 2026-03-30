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

/// Cell dimensions (characters).
///
/// Each hex is 6 wide × 3 tall, rendered with half-block characters:
///
///    ▄▄▄▄        row 0  (narrower top)
///   ▐ XX ▌       row 1  (full-width middle)
///    ▀▀▀▀        row 2  (narrower bottom)
///
/// Axial → screen (top-left corner):
///   sx = cx + q * W + r * (W / 2)
///   sy = cy + r * H
const W: i32 = 6;
const H: i32 = 3;
const STATUS_H: u16 = 5;

struct App {
    game: GameState,
    cursor: Option<Coord>,
    message: String,
    config: GameConfig,
    /// Top-left of the (0,0) hex, set each frame during render.
    origin_x: i32,
    origin_y: i32,
}

impl App {
    fn new(config: GameConfig) -> Self {
        Self {
            game: GameState::with_config(config),
            cursor: None,
            message: "Click a hex to place a stone. Press 'q' to quit, 'r' to restart.".into(),
            config,
            origin_x: 0,
            origin_y: 0,
        }
    }

    fn restart(&mut self) {
        self.game = GameState::with_config(self.config);
        self.cursor = None;
        self.message = "New game started.".into();
    }

    /// Convert screen pixel (col, row) → axial (q, r).
    ///
    /// Uses cube-coordinate rounding for accurate hex hit-testing.
    fn screen_to_axial(&self, col: u16, row: u16) -> Coord {
        // Centre of the (0,0) hex in screen space.
        let ccx = self.origin_x as f64 + W as f64 / 2.0;
        let ccy = self.origin_y as f64 + H as f64 / 2.0;

        let dx = col as f64 - ccx;
        let dy = row as f64 - ccy;

        // Fractional axial coords (inverse of the forward mapping).
        let rf = dy / H as f64;
        let qf = (dx - rf * W as f64 / 2.0) / W as f64;

        // Cube-coordinate rounding: convert axial → cube, round, fix constraint.
        let cx = qf;
        let cz = rf;
        let cy = -cx - cz;

        let mut rx = cx.round();
        let ry = cy.round();
        let mut rz = cz.round();

        let ex = (rx - cx).abs();
        let ey = (ry - cy).abs();
        let ez = (rz - cz).abs();

        if ex > ey && ex > ez {
            rx = -ry - rz;
        } else if ey > ez {
            // ry not needed for output, but fix for correctness
            let _ = -rx - rz;
        } else {
            rz = -rx - ry;
        }

        (rx as i32, rz as i32)
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
                            self.message = format!(
                                "{} wins! ({stones} stones). Press 'r' to restart.",
                                sym(p)
                            );
                        }
                        None => {
                            self.message =
                                format!("Draw after {stones} stones. Press 'r' to restart.");
                        }
                    }
                } else {
                    let p = self.game.current_player().unwrap();
                    let remaining = self.game.moves_remaining_this_turn();
                    self.message = format!(
                        "{} to move ({remaining} left). Placed at ({}, {}). {stones} stones.",
                        sym(p),
                        coord.0,
                        coord.1,
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

fn sym(p: Player) -> &'static str {
    match p {
        Player::P1 => "X",
        Player::P2 => "O",
    }
}

fn axial_to_screen(q: i32, r: i32, ox: i32, oy: i32) -> (i32, i32) {
    (ox + q * W + r * (W / 2), oy + r * H)
}

fn in_board(sx: i32, sy: i32, area: Rect) -> bool {
    sx >= area.x as i32
        && sx + W - 1 < (area.x + area.width) as i32
        && sy >= area.y as i32
        && sy + H - 1 < (area.y + area.height) as i32
}

/// Draw a half-block hex cell:
///
///   ▄▄▄▄        (fg = color, narrower top)
///  ▐ XX ▌       (fg = color, full-width sides + content)
///   ▀▀▀▀        (fg = color, narrower bottom)
fn draw_hex(frame: &mut Frame, sx: i32, sy: i32, content: &str, color: Color, bold: bool) {
    let border = Style::default().fg(color);

    // Row 0 — top: space + 4 lower-half-blocks + space
    frame.render_widget(
        Span::styled(" \u{2584}\u{2584}\u{2584}\u{2584} ", border),
        Rect::new(sx as u16, sy as u16, W as u16, 1),
    );

    // Row 1 — middle: left-half-block + content + right-half-block
    let cs = if bold {
        Style::default().fg(color).bold()
    } else {
        border
    };
    let line = Line::from(vec![
        Span::styled("\u{2590}", border),  // ▐
        Span::styled(content, cs),
        Span::styled("\u{258c}", border),  // ▌
    ]);
    frame.render_widget(line, Rect::new(sx as u16, (sy + 1) as u16, W as u16, 1));

    // Row 2 — bottom: space + 4 upper-half-blocks + space
    frame.render_widget(
        Span::styled(" \u{2580}\u{2580}\u{2580}\u{2580} ", border),
        Rect::new(sx as u16, (sy + 2) as u16, W as u16, 1),
    );
}

fn render(frame: &mut Frame, app: &mut App) {
    let area = frame.area();

    let [board_area, status_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(STATUS_H)]).areas(area);

    // The (0,0) hex is centred in the board area.
    let ox = board_area.x as i32 + board_area.width as i32 / 2 - W / 2;
    let oy = board_area.y as i32 + board_area.height as i32 / 2 - H / 2;
    app.origin_x = ox;
    app.origin_y = oy;

    let stones = app.game.placed_stones();
    let legal = app.game.legal_moves_set();

    // 1. Empty hex outlines for nearby legal moves.
    if !app.game.is_terminal() {
        for &(q, r) in legal.iter() {
            let (sx, sy) = axial_to_screen(q, r, ox, oy);
            if !in_board(sx, sy, board_area) {
                continue;
            }
            let near = stones
                .iter()
                .any(|&(c, _)| hexo_rs::hex::hex_distance((q, r), c) <= 2);
            if !near {
                continue;
            }
            draw_hex(frame, sx, sy, "    ", Color::Indexed(236), false);
        }
    }

    // 2. Placed stones.
    for &((q, r), player) in &stones {
        let (sx, sy) = axial_to_screen(q, r, ox, oy);
        if !in_board(sx, sy, board_area) {
            continue;
        }
        let (label, color) = match player {
            Player::P1 => (" XX ", Color::Cyan),
            Player::P2 => (" OO ", Color::Magenta),
        };
        draw_hex(frame, sx, sy, label, color, true);
    }

    // 3. Cursor highlight.
    if let Some((q, r)) = app.cursor {
        let (sx, sy) = axial_to_screen(q, r, ox, oy);
        if in_board(sx, sy, board_area) {
            let is_stone = stones.iter().any(|&(c, _)| c == (q, r));
            let is_legal = legal.contains(&(q, r));

            if is_stone {
                // Bright brackets on the middle row.
                let hl = Style::default().fg(Color::Yellow).bold();
                frame.render_widget(
                    Span::styled("\u{25b8}", hl), // ▸
                    Rect::new(sx as u16, (sy + 1) as u16, 1, 1),
                );
                frame.render_widget(
                    Span::styled("\u{25c2}", hl), // ◂
                    Rect::new((sx + W - 1) as u16, (sy + 1) as u16, 1, 1),
                );
            } else if is_legal {
                draw_hex(frame, sx, sy, " \u{b7}\u{b7} ", Color::Yellow, true);
            } else {
                draw_hex(frame, sx, sy, "    ", Color::Indexed(240), false);
            }
        }
    }

    // Status bar.
    let player_info = if let Some(p) = app.game.current_player() {
        let remaining = app.game.moves_remaining_this_turn();
        format!(
            " {} to move \u{2502} {remaining} moves left \u{2502} {} stones ",
            sym(p),
            app.game.placed_stones().len()
        )
    } else {
        match app.game.winner() {
            Some(p) => format!(" {} wins! ", sym(p)),
            None => " Draw! ".into(),
        }
    };

    let status = Paragraph::new(vec![
        Line::from(app.message.as_str()),
        Line::from(player_info),
        Line::from(" q: quit \u{2502} r: restart \u{2502} click: place stone "),
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
        terminal.draw(|frame| render(frame, &mut app))?;

        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('r') => app.restart(),
                    _ => {}
                },
                Event::Mouse(mouse) => match mouse.kind {
                    MouseEventKind::Moved | MouseEventKind::Drag(MouseButton::Left) => {
                        app.cursor = Some(app.screen_to_axial(mouse.column, mouse.row));
                    }
                    MouseEventKind::Down(MouseButton::Left) => {
                        let coord = app.screen_to_axial(mouse.column, mouse.row);
                        app.try_place(coord);
                        app.cursor = Some(coord);
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    ratatui::restore();
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;
    disable_raw_mode()?;
    Ok(())
}
