-- Data loader for training of ED models.

train_file = opt.root_data_dir .. 'generated/test_train_data/aida_train.csv'
it_train, _ = io.open(train_file)
type_file = opt.root_data_dir .. 'generated/test_train_data/aida_train_type.csv'
it_type, _ = io.open(type_file)

print('==> Loading training data with option ' .. opt.store_train_data)
--one doc, one minibatch
local function one_doc_to_minibatch(doc_lines, doc_type_lines)
  -- Create empty mini batch:
  assert(#doc_lines == #doc_type_lines)
  local num_mentions = #doc_lines
  assert(num_mentions > 0)

  local inputs = empty_minibatch_with_ids(num_mentions) -- initialize
  local targets = torch.zeros(num_mentions)

  -- Fill in each example:
  for i = 1, num_mentions do
    local target = process_one_line(doc_lines[i], doc_type_lines[i], inputs, i, true)
    targets[i] = target
    assert(target >= 1 and target == targets[i])
  end

  return inputs, targets  
end

if opt.store_train_data == 'RAM' then
  -- all_docs_inputs = tds.Hash()
  all_docs_inputs = {}
  all_docs_targets = tds.Hash()
  doc2id = tds.Hash()
  id2doc = tds.Hash()

  local cur_doc_lines = tds.Hash()
  local cur_doc_type_lines = tds.Hash()
  local prev_doc_id = nil

  local line = it_train:read()
  local type_line = it_type:read()
  while line do
    local parts = split(line, '\t')
    local type_parts = split(type_line, '\t')
    assert(parts[1] == type_parts[1], parts[1] .. '\t' .. type_parts[1])
    local doc_name = parts[1]
    if not doc2id[doc_name] then --all previous doc information has been loaded
      if prev_doc_id then
        local inputs, targets = one_doc_to_minibatch(cur_doc_lines, cur_doc_type_lines)
        -- print(inputs)
        -- all_docs_inputs[prev_doc_id] = minibatch_table2tds(inputs)
        all_docs_inputs[prev_doc_id] = inputs
        all_docs_targets[prev_doc_id] = targets
      end
      local cur_docid = 1 + #doc2id
      id2doc[cur_docid] = doc_name
      doc2id[doc_name] = cur_docid
      cur_doc_lines = tds.Hash()
      cur_doc_type_lines = tds.Hash()
      prev_doc_id = cur_docid
    end
    cur_doc_lines[1 + #cur_doc_lines] = line
    cur_doc_type_lines[1 + #cur_doc_type_lines] = type_line
    line = it_train:read()
    type_line = it_type:read()
  end
  if prev_doc_id then
    local inputs, targets = one_doc_to_minibatch(cur_doc_lines, cur_doc_type_lines)
    -- all_docs_inputs[prev_doc_id] = minibatch_table2tds(inputs)
    all_docs_inputs[prev_doc_id] = inputs
    all_docs_targets[prev_doc_id] = targets
  end  
  assert(#doc2id == #all_docs_inputs, #doc2id .. ' ' .. #all_docs_inputs)

else
  all_doc_lines = tds.Hash()
  all_doc_type_lines = tds.Hash()
  doc2id = tds.Hash()
  id2doc = tds.Hash()

  local line = it_train:read()
  local type_line = it_type:read()
  while line do
    local parts = split(line, '\t')
    local type_parts = split(type_line, '\t')
    local doc_name = parts[1]
    assert(doc_name == type_parts[1])
    if not doc2id[doc_name] then
      local cur_docid = 1 + #doc2id
      id2doc[cur_docid] = doc_name
      doc2id[doc_name] = cur_docid
      all_doc_lines[cur_docid] = tds.Hash()
      all_doc_type_lines[cur_docid] = tds.Hash()
    end
    all_doc_lines[doc2id[doc_name]][1 + #all_doc_lines[doc2id[doc_name]]] = line
    all_doc_type_lines[doc2id[doc_name]][1 + #all_doc_type_lines[doc2id[doc_name]]] = type_line
    line = it_train:read()
    type_line = it_type:read()
  end
  assert(#doc2id == #all_doc_lines)
  assert(#doc2id == #all_doc_type_lines)
end


get_minibatch = function()
  -- Create empty mini batch:
  local inputs = nil
  local targets = nil

  if opt.store_train_data == 'RAM' then
    local random_docid = math.random(#id2doc)
    -- inputs = minibatch_tds2table(all_docs_inputs[random_docid])
    inputs = deep_copy(all_docs_inputs[random_docid])
    targets = all_docs_targets[random_docid]
    -- print("doc id" .. '\t' ..  random_docid)
  else
    local random_docid = math.random(#id2doc)
    local doc_lines = all_doc_lines[random_docid]
    local doc_type_lines = all_doc_type_lines[random_docid]
    inputs, targets = one_doc_to_minibatch(doc_lines, doc_type_lines)
  end

  -- Move data to GPU:
  -- print("BEFORE" .. '\n')
  -- print(inputs)
  -- print(inputs[5][1])
  inputs, targets = minibatch_to_correct_type(inputs, targets, true)
  targets = correct_type(targets)

  return inputs, targets
end

print('    Done loading training data.')
