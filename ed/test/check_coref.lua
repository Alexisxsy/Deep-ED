-- Runs our trivial coreference resolution method and outputs the new set of 
-- entity candidates. Used for debugging the coreference resolution method.

if not opt then
  cmd = torch.CmdLine()
  cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')
  cmd:text()
  opt = cmd:parse(arg or {})
  assert(opt.root_data_dir ~= '', 'Specify a valid root_data_dir path argument.')  
end


require 'torch'
dofile 'utils/utils.lua'

tds = tds or require 'tds'

dofile 'entities/ent_name2id_freq/ent_name_id.lua'
dofile 'ed/test/coref_persons.lua'

file = opt.root_data_dir .. 'generated/test_train_data/aida_testA.csv'
file_type = opt.root_data_dir .. 'generated/test_train_data/aida_testA_type.csv'

opt = {}
opt.coref = true

  it, _ = io.open(file)
  it_type, _ = io.open(file_type)
  local all_doc_lines = tds.Hash()
  local all_doc_type_lines = tds.Hash()
  local line = it:read()
  local type_line = it_type:read()
  while line do
    local parts = split(line, '\t')
    local type_parts = split(type_line, '\t')
    local doc_name = parts[1]
    assert(type_parts[1] == parts[1])
    if not all_doc_lines[doc_name] then
      assert(not all_doc_type_lines[doc_name])
      all_doc_lines[doc_name] = tds.Hash()
      all_doc_type_lines[doc_name] = tds.Hash()
    end
    all_doc_lines[doc_name][1 + #all_doc_lines[doc_name]] = line
    all_doc_type_lines[doc_name][1 + #all_doc_type_lines[doc_name]] = type_line
    line = it:read()
    type_line = it_type:read()
  end
  -- Gather coreferent mentions to increase accuracy.
  build_coreference_dataset(all_doc_lines, all_doc_type_lines, 'aida-A')
