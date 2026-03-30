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

/// Cell: 4 wide × 2 tall, half-block characters.
///
///   ▄▄        row 0  (top cap, inset by 1)
///  ▐X ▌       row 1  (sides + content)
///   ▀▀        row 2  (bottom cap, inset by 1... wait, that's 3 rows)
///
/// Actually 4 wide × 3 tall for a nice hex shape:
///  ▄▄
/// ▐  ▌
///  ▀▀
const W: i32 = 4;
const H: i32 = 3;
const STATUS_H: u16 = 5;

struct App {
    game: GameState,
    cursor: Option<Coord>,
    message: String,
    config: GameConfig,
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

    fn screen_to_axial(&self, col: u16, row: u16) -> Coord {
        let ccx = self.origin_x as f64 + W as f64 / 2.0;
        let ccy = self.origin_y as f64 + H as f64 / 2.0;

        let dx = col as f64 - ccx;
        let dy = row as f64 - ccy;

        let rf = dy / H as f64;
        let qf = (dx - rf * W as f64 / 2.0) / W as f64;

        // Cube-coordinate rounding.
        let cube_x = qf;
        let cube_z = rf;
        let cube_y = -cube_x - cube_z;

        let mut rx = cube_x.round();
        let ry = cube_y.round();
        let mut rz = cube_z.round();

        let ex = (rx - cube_x).abs();
        let ey = (ry - cube_y).abs();
        let ez = (rz - cube_z).abs();

        if ex > ey && ex > ez {
            rx = -ry - rz;
        } else if ey <= ez {
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

/// Minimum distance from `coord` to any placed stone.
fn min_stone_dist(coord: Coord, stones: &[(Coord, Player)]) -> i32 {
    stones
        .iter()
        .map(|&(c, _)| hexo_rs::hex::hex_distance(coord, c))
        .min()
        .unwrap_or(i32::MAX)
}

/// Map a hex distance to a style that fades with distance.
/// Uses DarkGray color with varying modifiers — works on both light and dark terminals.
fn fade_style(dist: i32, max_dist: i32) -> Option<Style> {
    if dist > max_dist {
        return None;
    }
    let t = (dist - 1).max(0) as f64 / (max_dist - 1).max(1) as f64; // 0.0 (close) .. 1.0 (far)
    if t < 0.33 {
        // Close: normal brightness
        Some(Style::default().fg(Color::Gray))
    } else if t < 0.66 {
        // Mid: darker
        Some(Style::default().fg(Color::DarkGray))
    } else {
        // Far: dim
        Some(Style::default().fg(Color::DarkGray).dim())
    }
}

/// Draw a 4×3 half-block hex with explicit style.
fn draw_hex_styled(frame: &mut Frame, sx: i32, sy: i32, ch: &str, style: Style) {
    frame.render_widget(
        Span::styled(" \u{2584}\u{2584} ", style),
        Rect::new(sx as u16, sy as u16, W as u16, 1),
    );
    let line = Line::from(vec![
        Span::styled("\u{2590}", style),  // ▐
        Span::styled(ch, style),
        Span::styled("\u{258c}", style),  // ▌
    ]);
    frame.render_widget(line, Rect::new(sx as u16, (sy + 1) as u16, W as u16, 1));
    frame.render_widget(
        Span::styled(" \u{2580}\u{2580} ", style),
        Rect::new(sx as u16, (sy + 2) as u16, W as u16, 1),
    );
}

fn render(frame: &mut Frame, app: &mut App) {
    let area = frame.area();

    let [board_area, status_area] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(STATUS_H)]).areas(area);

    let ox = board_area.x as i32 + board_area.width as i32 / 2 - W / 2;
    let oy = board_area.y as i32 + board_area.height as i32 / 2 - H / 2;
    app.origin_x = ox;
    app.origin_y = oy;

    let stones = app.game.placed_stones();
    let legal = app.game.legal_moves_set();
    let radius = app.config.placement_radius;

    // 1. Empty hex outlines with distance-based fade.
    if !app.game.is_terminal() {
        for &(q, r) in legal.iter() {
            let (sx, sy) = axial_to_screen(q, r, ox, oy);
            if !in_board(sx, sy, board_area) {
                continue;
            }
            let dist = min_stone_dist((q, r), &stones);
            if let Some(style) = fade_style(dist, radius) {
                draw_hex_styled(frame, sx, sy, "  ", style);
            }
        }
    }

    // 2. Placed stones.
    for &((q, r), player) in &stones {
        let (sx, sy) = axial_to_screen(q, r, ox, oy);
        if !in_board(sx, sy, board_area) {
            continue;
        }
        let (ch, style) = match player {
            Player::P1 => ("X ", Style::default().fg(Color::Cyan).bold()),
            Player::P2 => ("O ", Style::default().fg(Color::Magenta).bold()),
        };
        draw_hex_styled(frame, sx, sy, ch, style);
    }

    // 3. Cursor highlight.
    if let Some((q, r)) = app.cursor {
        let (sx, sy) = axial_to_screen(q, r, ox, oy);
        if in_board(sx, sy, board_area) {
            let is_stone = stones.iter().any(|&(c, _)| c == (q, r));
            let is_legal = legal.contains(&(q, r));

            if is_stone {
                let hl = Style::default().fg(Color::Yellow).bold();
                frame.render_widget(
                    Span::styled("\u{25b8}", hl),
                    Rect::new(sx as u16, (sy + 1) as u16, 1, 1),
                );
                frame.render_widget(
                    Span::styled("\u{25c2}", hl),
                    Rect::new((sx + W - 1) as u16, (sy + 1) as u16, 1, 1),
                );
            } else if is_legal {
                draw_hex_styled(frame, sx, sy, "\u{b7}\u{b7}", Style::default().fg(Color::Yellow).bold());
            } else {
                draw_hex_styled(frame, sx, sy, "  ", Style::default().fg(Color::DarkGray).dim());
            }
        }
    }

    // Status bar.
    let player_info = if let Some(p) = app.game.current_player() {
        let remaining = app.game.moves_remaining_this_turn();
        format!(
            " {} to move \u{2502} {remaining} left \u{2502} {} stones ",
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
        Line::from(" q: quit \u{2502} r: restart \u{2502} click: place "),
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
