#include "logger.hpp"

#include <fmt/chrono.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <string>

namespace tools
{
std::shared_ptr<spdlog::logger> logger_ = nullptr;

void set_logger()
{
  auto file_name = fmt::format("logs/{:%Y-%m-%d_%H-%M-%S}.log", std::chrono::system_clock::now());
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(file_name, true);
  file_sink->set_level(spdlog::level::debug);

  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  // 控制台只放 info 以上: debug 级是高频路径(丢失/发散/弹道失败), 刷屏会淹没
  // 真正要看的仪表行。完整 debug 日志仍写入文件 sink, 诊断能力不受损。
  console_sink->set_level(spdlog::level::info);

  logger_ = std::make_shared<spdlog::logger>("", spdlog::sinks_init_list{file_sink, console_sink});
  logger_->set_level(spdlog::level::debug);
  logger_->flush_on(spdlog::level::info);
}

std::shared_ptr<spdlog::logger> logger()
{
  if (!logger_) set_logger();
  return logger_;
}

}  // namespace tools
