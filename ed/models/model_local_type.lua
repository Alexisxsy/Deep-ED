----- Define the model
function local_type_model(num_mentions, param_T_linear)
    assert(num_mentions)
    assert(param_T_linear)

    model = nn.Sequential()
    
    local ctxt_type_vec_and_ent_lookup = nn.ConcatTable()
        :add(nn.Sequential()
            :add(nn.SelectTable(4))    -- 4: mention type, num_mention * type_num 
            :add(nn.View(num_mentions, opt.num_type))
        )
        :add(nn.Sequential()
            :add(nn.SelectTable(5))    -- 5: entity type, num_mention * max_num_cand * type_num   
            :add(nn.View(num_mentions, max_num_cand, opt.num_type))
        )
    
    model:add(ctxt_type_vec_and_ent_lookup)

    local entity_ctx_type_sim_scores = nn.Sequential()
        :add(nn.ConcatTable()
            :add(nn.SelectTable(2))
            :add(nn.Sequential()
                :add(nn.SelectTable(1))
                :add(param_T_linear)
                :add(nn.View(num_mentions, opt.num_type, 1))
            ) 
        )
        :add(nn.MM())
        :add(nn.View(num_mentions, max_num_cand))

    model:add(entity_ctx_type_sim_scores)

    if string.find(opt.type, 'cuda') then
        model = model:cuda()
    end

    return model

end