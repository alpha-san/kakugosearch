use anyhow::Result;
use tracing_subscriber::EnvFilter;

mod api;
mod ai;
mod config;
mod index;
mod vector;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("kakugosearch=info".parse()?))
        .init();

    let config = config::AppConfig::load("kakugosearch.toml")?;
    tracing::info!("Starting KakugoSearch v{}", env!("CARGO_PKG_VERSION"));

    let app_state = api::AppState::new(&config).await?;
    let app = api::router(app_state);

    let addr = format!("{}:{}", config.server.host, config.server.port);
    tracing::info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
