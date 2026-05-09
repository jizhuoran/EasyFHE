#include <ATen/cuda/tunable/Tunable.h>

namespace at::cuda::tunable {

std::ostream& operator<<(std::ostream& stream, const ResultEntry& entry) {
  return stream << entry.GetKey();
}

KernelMap TuningResultsManager::Lookup(const std::string&) {
  return {};
}

ResultEntry TuningResultsManager::Lookup(const std::string&, const std::string&) {
  return ResultEntry::Null();
}

void TuningResultsManager::AddImpl(const std::string&, const std::string&, ResultEntry, KernelMap&) {}
void TuningResultsManager::Add(const std::string&, const std::string&, ResultEntry) {}
void TuningResultsManager::Delete(const std::string&, const std::string&) {}
void TuningResultsManager::DisjointMergeImpl(const std::string&, const KernelMap&, ResultsMap&) {}
void TuningResultsManager::Load(const ResultsMap&) {}

ResultsMap TuningResultsManager::Dump() {
  return {};
}

void TuningResultsManager::DisjointMerge(const std::string&, const KernelMap&) {}

size_t TuningResultsManager::GetSize() {
  return 0;
}

void TuningResultsManager::RecordUntuned(
    std::ofstream&,
    const std::string&,
    const std::string&,
    const std::string&) {}

void TuningResultsManager::InitRealtimeAppend(
    const std::string&,
    const std::unordered_map<std::string, std::string>&) {}

void TuningResultsManager::AppendResultLine(
    const std::string&,
    const std::string&,
    const ResultEntry&) {}

void TuningResultsManager::CloseRealtimeAppend() {}

TuningResultsValidator::TuningResultsValidator() = default;

std::unordered_map<std::string, std::string> TuningResultsValidator::GetAllValidators() const {
  return {};
}

TuningStatus TuningResultsValidator::ValidateAll(
    const std::unordered_map<std::string, std::string>&) const {
  return OK;
}

void TuningResultsValidator::RegisterValidator(
    const std::string&,
    const GetFunc&,
    const ValidateFunc&) {}

std::string TuningResultsValidator::GetPyTorchVersion() {
  return {};
}

TuningStatus TuningResultsValidator::ValidatePyTorchVersion(const std::string&) const {
  return OK;
}

TuningContext::TuningContext()
    : enable_(false),
      tuning_enable_(false),
      record_untuned_enable_(false),
      manager_initialized_(false),
      numerics_check_enable_(false),
      max_tuning_duration_ms_(0),
      max_tuning_iterations_(0),
      max_warmup_duration_ms_(0),
      max_warmup_iterations_(0),
      icache_flush_(false),
      rotating_buffer_size_(0),
      results_count_from_input_file_(0),
      is_shutting_down_(false) {}

TuningContext::~TuningContext() = default;

void TuningContext::EnableTunableOp(bool) {
  enable_ = false;
}

bool TuningContext::IsTunableOpEnabled() const {
  return false;
}

void TuningContext::EnableTuning(bool) {
  tuning_enable_ = false;
}

bool TuningContext::IsTuningEnabled() const {
  return false;
}

void TuningContext::EnableRecordUntuned(bool) {
  record_untuned_enable_ = false;
}

bool TuningContext::IsRecordUntunedEnabled() const {
  return false;
}

std::ofstream& TuningContext::GetUntunedFile() {
  return untuned_file_;
}

void TuningContext::EnableNumericsCheck(bool value) {
  numerics_check_enable_ = value;
  numerics_cfg_.enabled = value;
}

bool TuningContext::IsNumericsCheckEnabled() const {
  return numerics_check_enable_;
}

void TuningContext::SetNumericalCheckConfig(bool enabled, double atol, double rtol) {
  numerics_check_enable_ = enabled;
  numerics_cfg_ = NumericalCheckConfig(enabled, atol, rtol);
}

NumericalCheckConfig TuningContext::GetNumericalCheckConfig() const {
  return numerics_cfg_;
}

void TuningContext::SetMaxTuningDurationMs(int max_duration_ms) {
  max_tuning_duration_ms_ = max_duration_ms;
}

int TuningContext::GetMaxTuningDurationMs() const {
  return max_tuning_duration_ms_;
}

void TuningContext::SetMaxTuningIterations(int max_iter) {
  max_tuning_iterations_ = max_iter;
}

int TuningContext::GetMaxTuningIterations() const {
  return max_tuning_iterations_;
}

void TuningContext::SetMaxWarmupDurationMs(int max_duration_ms) {
  max_warmup_duration_ms_ = max_duration_ms;
}

int TuningContext::GetMaxWarmupDurationMs() const {
  return max_warmup_duration_ms_;
}

void TuningContext::SetMaxWarmupIterations(int max_iter) {
  max_warmup_iterations_ = max_iter;
}

int TuningContext::GetMaxWarmupIterations() const {
  return max_warmup_iterations_;
}

void TuningContext::EnableICacheFlush(bool value) {
  icache_flush_ = value;
}

bool TuningContext::IsICacheFlushEnabled() const {
  return icache_flush_;
}

void TuningContext::SetRotatingBufferSize(int size) {
  rotating_buffer_size_ = size;
}

int TuningContext::GetRotatingBufferSize() const {
  return rotating_buffer_size_;
}

TuningResultsManager& TuningContext::GetTuningResultsManager() {
  return manager_;
}

TuningResultsValidator& TuningContext::GetTuningResultsValidator() {
  return validator_;
}

TuningResults TuningContext::GetTuningResults() {
  return {};
}

TuningStatus TuningContext::LoadTuningResults(const TuningResults&) {
  return OK;
}

void TuningContext::SetFilename(const std::string& filename, bool) {
  filename_ = filename;
}

std::string TuningContext::GetFilename() const {
  return filename_;
}

bool TuningContext::ReadFile(const std::string&) {
  return false;
}

std::string TuningContext::GetLogFilename() const {
  return {};
}

int TuningContext::GetLogLevel() const {
  return 0;
}

bool TuningContext::GetLogOkay() const {
  return false;
}

std::ostream& TuningContext::GetLog() const {
  return std::clog;
}

TuningContext* getTuningContext() {
  static TuningContext context;
  return &context;
}

} // namespace at::cuda::tunable
