#include <fstream>
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

namespace nnvm {
namespace pass {
namespace {

struct SA_Node {
  uint32_t sa_nid;
  std::string name;
  uint32_t tensor_nid;
  uint32_t tensor_idx;
  std::vector<uint32_t> deps;
  std::vector<std::pair<uint32_t, uint32_t>> inputs;
  std::unordered_set<uint32_t> be_depended;
};

void LoadHandleUsages(uint32_t nid, std::string& line,
                      HandleUsages& hdl_usages,
                      HandleSizes& hdl_sizes) {
  //std::cout << "LoadHandleUsages:" << line << std::endl;
  if (line.size() == 0) return;
  size_t next = 0, last = 0;
  next = line.find(",", last);
  while ((next = line.find(",", last)) != std::string::npos) {
    uint32_t hid = std::stoi(line.substr(last, next - last));
    hdl_usages[nid].push_back(hid);
    last = next + 1;
    next = line.find(",", last);
    size_t size = std::stol(line.substr(last, next - last));
    hdl_sizes[hid] = size;
    last = next + 1;
  }
}

void LoadInputDep(SA_Node& node, std::string& line) {
  //std::cout << "LoadInputDep:" << line << std::endl;
  if (line.size() == 0) return;
  size_t next = 0, last = 0;
  while ((next = line.find(",", last)) != std::string::npos) {
    uint32_t node_id = std::stoi(line.substr(last, next - last));
    last = next + 1;
    next = line.find(",", last);
    uint32_t index = std::stoi(line.substr(last, next - last));
    last = next + 1;
    node.inputs.push_back(std::make_pair(node_id, index));
  }
}

uint32_t LoadNodeInfo(std::string& line,
                      std::unordered_map<uint32_t, SA_Node>& sa_nodes) {
  //std::cout << "LoadNodeInfo:" << line << std::endl;
  size_t next = 0, last = 0;
  next = line.find(",", last);
  size_t sa_nid = std::stoi(line.substr(last, next - last));
  sa_nodes[sa_nid].sa_nid = sa_nid;
  SA_Node& node = sa_nodes[sa_nid];
  last = next + 1;
  next = line.find(",", last);
  node.name = line.substr(last, next - last);
  last = next + 1;
  next = line.find(",", last);
  std::string nid = line.substr(last, next - last);
  last = next + 1;
  next = line.find(",", last);
  std::string idx = line.substr(last, next - last);
  if (nid[0] == 'N') {
    node.tensor_nid = -1;
    node.tensor_idx = -1;
  } else {
    node.tensor_nid = std::stoi(nid);
    node.tensor_idx = std::stoi(idx);
  }
  last = next + 1;
  while ((next = line.find(",", last)) != std::string::npos) {
    node.deps.push_back(std::stoi(line.substr(last, next - last)));
    last = next + 1;
  }
  return sa_nid;
}

void LoadSAGraphFile(std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                     HandleUsages& handle_usages, HandleSizes& handle_sizes) {
  //std::cout << "LoadSAGraphFile" << std::endl;
  std::ifstream ifs("dataflow.rst");
  std::string line, node_info, hdl_usages, input_deps;
  while (std::getline(ifs, line)) {
    size_t next = 0, last = 0;
    next = line.find(";", last);
    node_info = line.substr(last, next - last);
    last = next + 1;
    next = line.find(";", last);
    hdl_usages = line.substr(last, next - last);
    input_deps = line.substr(next + 1);

    uint32_t sa_nid = LoadNodeInfo(node_info, sa_nodes);
    SA_Node& node = sa_nodes[sa_nid];
    LoadHandleUsages(sa_nid, hdl_usages, handle_usages, handle_sizes);
    LoadInputDep(node, input_deps);
  }
  //std::cout << "SA_Node count " << sa_nodes.size() << std::endl;
}

NodeEntry CreateSwapEntry(const Op* swap_source_op) {
  std::cout << "CreateSwapEntry" << std::endl;
  NodePtr node = Node::Create();
  node->attrs.op = swap_source_op;
  node->attrs.name = "swap_entry";
  std::ostringstream os;
  os << "_SwapEntry_var";
  // Note(fegin): We don't create a new variable for SwapEntry's input.
  // Instead, we use "data" as the input for SwapEntry. This simplier the
  // debuging of MXNetcreating data_entry_.
  //node->inputs.emplace_back(Symbol::CreateVariable("_SwapEntry_var").outputs[0]);
  return NodeEntry{std::move(node), 0, 0};
}

NodeEntry CreateSwapoutSink(const Op* swapout_sink_op) {
  //std::cout << "CreateSwapoutSink" << std::endl;
  NodePtr node = Node::Create();
  node->attrs.op = swapout_sink_op;
  node->attrs.name = "swapout_sink";
  return NodeEntry{std::move(node), 0, 0};
}

void CreateSwapout(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                   const NodeEntry& swap_entry,
                   const NodeEntry& swapout_sink,
                   const Op* swapout_op,
                   std::unordered_map<uint32_t, NodeEntry>& swapouts) {
  //std::cout << "CreateSwapout" << std::endl;
  for (const auto& kv: sa_nodes) {
    if (kv.second.name != "swapout") continue;
    NodePtr node = Node::Create();
    node->attrs.op = swapout_op;
    node->attrs.name = "swapout_" + std::to_string(kv.first);
    node->attrs.dict["src_tensor_nid"] = std::to_string(kv.second.tensor_nid);
    node->attrs.dict["src_tensor_idx"] = std::to_string(kv.second.tensor_idx);
    node->attrs.op->attr_parser(&(node->attrs));
    node->control_deps.emplace_back(swap_entry.node);
    swapout_sink.node->control_deps.emplace_back(node);
    swapouts[kv.first] = NodeEntry{std::move(node), 0, 0};
  }
}

void CreateSwapin(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                  const NodeEntry& swap_entry,
                  const Op* swapin_op,
                  std::unordered_map<uint32_t, NodeEntry>& swapins) {
  //std::cout << "CreateSwapin" << std::endl;
  for (const auto& kv: sa_nodes) {
    if (kv.second.name != "swapin") continue;
    NodePtr node = Node::Create();
    node->attrs.op = swapin_op;
    node->attrs.name = "swapin_" + std::to_string(kv.first);
    node->attrs.dict["src_tensor_nid"] = std::to_string(kv.second.tensor_nid);
    node->attrs.dict["src_tensor_idx"] = std::to_string(kv.second.tensor_idx);
    node->attrs.op->attr_parser(&(node->attrs));
    node->control_deps.emplace_back(swap_entry.node);
    swapins[kv.first] = NodeEntry{std::move(node), 0, 0};
  }
}

void CreateUpdate(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                  const NodeEntry& swapout_sink,
                  const Op* update_op,
                  std::unordered_map<uint32_t, NodeEntry>& updates) {
  //std::cout << "CreateUpdate" << std::endl;
  for (const auto& kv: sa_nodes) {
    if (kv.second.name.find("weight_g_update") == std::string::npos) continue;
    NodePtr node = Node::Create();
    node->attrs.op = update_op;
    node->attrs.name = kv.second.name;
    //node->attrs.op->attr_parser(&(node->attrs));
    swapout_sink.node->control_deps.emplace_back(node);
    updates[kv.first] = NodeEntry{std::move(node), 0, 0};
  }
}

void CreateVariables(std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                     const IndexedGraph& idx,
                     const NodeEntry& swap_entry,
                     std::unordered_map<uint32_t, NodeEntry>& variables,
                     std::unordered_map<Node*, uint32_t>& nodeptr_to_old_nid) {
  //std::cout << "CreateVariables" << std::endl;
  NodeEntry prev_var = {nullptr, 0, 0};
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    if (!idx[nid].source->is_variable()) continue;
    NodePtr node = nullptr;
    // FIXME(fegin): This is very hacky. Any better way to get NodePtr ?
    for (uint32_t dep_nid = 0; nid < idx.num_nodes(); ++dep_nid) {
      for (uint32_t control_idx = 0; control_idx < idx[nid].control_deps.size();
           ++control_idx) {
        if (idx[dep_nid].control_deps[control_idx] == nid) {
          node = idx[dep_nid].source->control_deps[control_idx];
          break;
        }
      }
      if (node != nullptr) {
        break;
      }
      for (uint32_t input_idx = 0; input_idx < idx[dep_nid].inputs.size();
           ++input_idx) {
        if (idx[dep_nid].inputs[input_idx].node_id == nid) {
          node = idx[dep_nid].source->inputs[input_idx].node;
          break;
        }
      }
      if (node != nullptr) {
        break;
      }
    }
    CHECK(node != nullptr);
    //LOG(INFO) << "Create variable " << node->attrs.name << std::endl;
    //LOG(INFO) << "Create variable " << sa_nodes.at(nid).name << std::endl;
    CHECK(node->attrs.name == sa_nodes.at(nid).name);
    nodeptr_to_old_nid[node.get()] = nid;
    if (prev_var.node != nullptr) {
      node->control_deps.emplace_back(prev_var.node);
    }
    if (node->attrs.name == "data") {
      variables[nid] = NodeEntry{std::move(node), 0, 0};
      swap_entry.node->inputs.emplace_back(variables[nid]);
    } else {
      node->control_deps.emplace_back(swap_entry.node);
      variables[nid] = NodeEntry{std::move(node), 0, 0};
    }
    prev_var = variables[nid];
    for (const auto dep_nid : sa_nodes.at(nid).deps) {
      sa_nodes.at(dep_nid).be_depended.insert(nid);
    }
  }
}

void CreateModelNodes(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                      const IndexedGraph& idx,
                      std::unordered_map<uint32_t, NodePtr>& new_nodes,
                      std::unordered_map<Node*, uint32_t>& nodeptr_to_old_nid) {
  //std::cout << "CreateModelNodes" << std::endl;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    if (idx[nid].source->is_variable()) continue;
    NodePtr new_node = Node::Create();
    new_node->attrs = idx[nid].source->attrs;
    //LOG(INFO) << "Create node " << new_node->attrs.name << std::endl;
    //LOG(INFO) << "Create node " << sa_nodes.at(nid).name << std::endl;
    CHECK(new_node->attrs.name == sa_nodes.at(nid).name);
    new_nodes[nid] = new_node;
    nodeptr_to_old_nid[new_node.get()] = nid;
  }
}

void ConnectSwapout(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                    const IndexedGraph& idx,
                    std::unordered_map<uint32_t, NodeEntry>& swapouts,
                    std::unordered_map<uint32_t, NodeEntry>& swapins,
                    std::unordered_map<uint32_t, NodeEntry>& updates,
                    std::unordered_map<uint32_t, NodeEntry>& variables,
                    std::unordered_map<uint32_t, NodePtr>& new_nodes) {
  //std::cout << "ConnectAllSwapout" << std::endl;
  for (auto& kv: swapouts) {
    uint32_t sa_nid = kv.first;
    if (sa_nodes.at(sa_nid).deps.size() == 0) continue;
    NodeEntry& entry = kv.second;
    for (uint32_t dep_nid : sa_nodes.at(sa_nid).deps) {
      // Depend on another swapout.
      auto so_it = swapouts.find(dep_nid);
      if (so_it != swapouts.end()) {
        entry.node->control_deps.emplace_back(so_it->second.node);
        continue;
      }

      // Depend on a swapin.
      auto si_it = swapins.find(dep_nid);
      if (si_it != swapins.end()) {
        entry.node->control_deps.emplace_back(si_it->second.node);
        continue;
      }

      // Depend on a variable.
      auto var_it = variables.find(dep_nid);
      if (var_it != variables.end()) {
        entry.node->control_deps.emplace_back(var_it->second.node);
        continue;
      }

      // Depend on a update node.
      auto update_it = updates.find(dep_nid);
      if (update_it != updates.end()) {
        // FIXME(fegin): Turn this on when everything is ready.
        entry.node->control_deps.emplace_back(update_it->second.node);
        continue;
      }

      // Depend on a model node.
      auto node_it = new_nodes.find(dep_nid);
      if (node_it != new_nodes.end()) {
        entry.node->control_deps.emplace_back(node_it->second);
        continue;
      }
      CHECK(false) << dep_nid  << ", " << new_nodes.size();
    }
  }
}

void ConnectPreSwapin(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                      const IndexedGraph& idx,
                      std::unordered_map<uint32_t, NodeEntry>& swapins,
                      std::unordered_map<uint32_t, NodeEntry>& variables) {
  // This function seems to be included by regular swapin functions.
  //std::cout << "ConnectPreSwapin" << std::endl;
  return;
  for (auto& kv: swapins) {
    uint32_t sa_nid = kv.first;
    if (sa_nodes.at(sa_nid).deps.size() > 0) continue;
    NodeEntry& entry = kv.second;
    for (const auto var_id : sa_nodes.at(sa_nid).be_depended) {
      //std::cout << " DEPENDS on PRESWAPIN" << std::endl;
      variables.at(var_id).node->control_deps.emplace_back(entry.node);
    }
  }
}

void ConnectAllSwapin(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                      const IndexedGraph& idx,
                      std::unordered_map<uint32_t, NodeEntry>& swapins,
                      std::unordered_map<uint32_t, NodeEntry>& swapouts,
                      std::unordered_map<uint32_t, NodeEntry>& updates,
                      std::unordered_map<uint32_t, NodeEntry>& variables,
                      std::unordered_map<uint32_t, NodePtr>& new_nodes) {
  //std::cout << "ConnectAllSwapin" << std::endl;
  for (auto& kv: swapins) {
    uint32_t sa_nid = kv.first;
    if (sa_nodes.at(sa_nid).deps.size() == 0) continue;
    NodeEntry& entry = kv.second;
    for (uint32_t dep_nid : sa_nodes.at(sa_nid).deps) {
      // Depend on another swapin.
      auto si_it = swapins.find(dep_nid);
      if (si_it != swapins.end()) {
        entry.node->control_deps.emplace_back(si_it->second.node);
        continue;
      }

      // Depend on a swapout.
      auto so_it = swapouts.find(dep_nid);
      if (so_it != swapouts.end()) {
        entry.node->control_deps.emplace_back(so_it->second.node);
        continue;
      }

      // Depend on a variable.
      auto var_it = variables.find(dep_nid);
      if (var_it != variables.end()) {
        entry.node->control_deps.emplace_back(var_it->second.node);
        continue;
      }

      // Depend on a update node.
      auto update_it = updates.find(dep_nid);
      if (update_it != updates.end()) {
        // FIXME(fegin): Turn this on when everything is ready.
        entry.node->control_deps.emplace_back(update_it->second.node);
        continue;
      }

      // Depend on a model node.
      auto node_it = new_nodes.find(dep_nid);
      if (node_it != new_nodes.end()) {
        entry.node->control_deps.emplace_back(node_it->second);
        continue;
      }
      CHECK(false) << sa_nid << ", " << dep_nid;
    }
  }
}

void ConnectUpdate(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                   const IndexedGraph& idx,
                   std::unordered_map<uint32_t, NodeEntry>& updates,
                   std::unordered_map<uint32_t, NodeEntry>& swapins,
                   std::unordered_map<uint32_t, NodeEntry>& variables,
                   std::unordered_map<uint32_t, NodePtr>& new_nodes) {
  //std::cout << "ConnectUpdate" << std::endl;
  for (auto& kv: updates) {
    const SA_Node& sa_node = sa_nodes.at(kv.first);
    NodeEntry& entry = kv.second;
    for (auto& input : sa_node.inputs) {
      //std::cout << "Input " << input.first << " " << input.second << std::endl;
      if (variables.count(input.first) == 1) {
          entry.node->inputs.emplace_back(variables.at(input.first));
      } else {
          entry.node->inputs.emplace_back(NodeEntry{new_nodes.at(input.first),
                                                    input.second,
                                                    0});
      }
    }
    if (sa_node.deps.size() == 0) continue;
    for (uint32_t dep_nid : sa_node.deps) {
      // Depend on a swapin.
      auto si_it = swapins.find(dep_nid);
      if (si_it != swapins.end()) {
        entry.node->control_deps.emplace_back(si_it->second.node);
        continue;
      }

      // Depend on a variable.
      auto var_it = variables.find(dep_nid);
      if (var_it != variables.end()) {
        entry.node->control_deps.emplace_back(var_it->second.node);
        continue;
      }

      // Depend on a model node.
      auto node_it = new_nodes.find(dep_nid);
      if (node_it != new_nodes.end()) {
        entry.node->control_deps.emplace_back(node_it->second);
        continue;
      }
      CHECK(false) << dep_nid  << ", " << new_nodes.size();
    }
  }
}
void ConnectModelNodes(const std::unordered_map<uint32_t, SA_Node>& sa_nodes,
                       const IndexedGraph& idx,
                       std::unordered_map<uint32_t, NodePtr>& new_nodes,
                       const std::unordered_map<uint32_t, NodeEntry>& swapouts,
                       const std::unordered_map<uint32_t, NodeEntry>& swapins,
                       std::unordered_map<uint32_t, NodeEntry>& variables) {
  //std::cout << "ConnectModelNodes" << std::endl;
  for (auto& kv : new_nodes) {
    uint32_t sa_nid = kv.first;
    auto old_inode = idx[sa_nid];
    NodePtr new_node = kv.second;

    // Copy inputs.
    for (const IndexedGraph::NodeEntry& ientry : old_inode.inputs) {
      auto var_it = variables.find(ientry.node_id);
      if (var_it != variables.end()) {
        new_node->inputs.emplace_back(NodeEntry{var_it->second.node,
                                                ientry.index,
                                                ientry.version});
      } else {
        new_node->inputs.emplace_back(NodeEntry{new_nodes.at(ientry.node_id),
                                                ientry.index,
                                                ientry.version});
      }
    }

    std::vector<uint32_t> deps;
    // Copy control nodes.
    for (uint32_t dep_nid : old_inode.control_deps) {
      deps.push_back(dep_nid);
    }
    // Control deps from SA
    const SA_Node& sa_node = sa_nodes.at(sa_nid);
    for (const uint32_t dep_nid : sa_node.deps) {
      bool exist = false;
      for (const uint32_t  _dep_nid : deps) {
        if (_dep_nid == dep_nid) {
          exist = true;
          break;
        }
      }
      if (!exist) {
        deps.push_back(dep_nid);
      }
    }

    //std::cout << std::endl;
    for (uint32_t dep_nid : deps) {
      //std::cout << "ConnectModeNode node " << new_node->attrs.name
                //<< " with " << dep_nid << std::endl;
      // Depend on another model node.
      auto node_it = new_nodes.find(dep_nid);
      if (node_it != new_nodes.end()) {
        //std::cout << "DEPENDS on another model node: "
                  //<< node_it->second->attrs.name << std::endl;
        new_node->control_deps.emplace_back(node_it->second);
        continue;
      }

      // Depend on a swapout.
      auto so_it = swapouts.find(dep_nid);
      if (so_it != swapouts.end()) {
        //std::cout << "DEPENDS on SWAPOUT" << std::endl;
        new_node->control_deps.emplace_back(so_it->second.node);
        continue;
      }

      // Depend on a swapin.
      auto si_it = swapins.find(dep_nid);
      if (si_it != swapins.end()) {
        //std::cout << "DEPENDS on SWAPIN" << std::endl;
        new_node->control_deps.emplace_back(si_it->second.node);
        continue;
      }

      // Depend on a variable.
      auto var_it = variables.find(dep_nid);
      if (var_it != variables.end()) {
        //std::cout << "DEPENDS on a variable." << std::endl;
        new_node->control_deps.emplace_back(var_it->second.node);
        continue;
      }

      CHECK(false) << dep_nid;
    }
  }
}

Graph SA_LoadGraph(Graph src) {
  std::cout << "SA_LoadGraph" << std::endl;
  CHECK(src.attrs.count("swap_entry_op"))
      << "Need graph attribute \"swap_entry_op\" in SA_LoadGraph";
  CHECK(src.attrs.count("swapout_sink_op"))
      << "Need graph attribute \"swapout_sink_op\" in SA_LoadGraph";
  CHECK(src.attrs.count("swapout_op"))
      << "Need graph attribute \"swapout_op\" in SA_LoadGraph";
  CHECK(src.attrs.count("swapin_op"))
      << "Need graph attribute \"swapin_op\" in SA_LoadGraph";
  CHECK(src.attrs.count("update_op"))
      << "Need graph attribute \"update_op\" in SA_LoadGraph";
  const Op* update_op = Op::Get(src.GetAttr<std::string>("update_op"));
  const Op* swap_entry_op = Op::Get(src.GetAttr<std::string>("swap_entry_op"));
  const Op* swapout_sink_op = Op::Get(src.GetAttr<std::string>("swapout_sink_op"));
  const Op* swapin_op = Op::Get(src.GetAttr<std::string>("swapin_op"));
  const Op* swapout_op = Op::Get(src.GetAttr<std::string>("swapout_op"));
  const IndexedGraph& idx = src.indexed_graph();
  std::unordered_map<Node*, uint32_t> nodeptr_to_old_nid;
  std::unordered_map<uint32_t, NodeEntry> swapouts;       // SA_ID -> swapout NodeEntry
  std::unordered_map<uint32_t, NodeEntry> swapins;        // SA_ID -> swapin NodeEntry
  std::unordered_map<uint32_t, NodeEntry> updates;        // SA_ID -> update NodeEntry
  std::unordered_map<uint32_t, NodeEntry> variables;      // SA_ID -> new variable NodeEntry
  std::unordered_map<uint32_t, NodePtr> new_nodes;        // SA_ID -> new model NodeEntry
  std::unordered_map<uint32_t, SA_Node> sa_nodes;         // SA_ID -> SA_Node
  HandleUsages handle_usages;
  HandleSizes handle_sizes;

  // Create all the new swapout, swapin and nodes.
  // Connect all of them together.
  LoadSAGraphFile(sa_nodes, handle_usages, handle_sizes);
  NodeEntry swap_entry = CreateSwapEntry(swap_entry_op);
  NodeEntry swapout_sink = CreateSwapoutSink(swapout_sink_op);
  CreateSwapout(sa_nodes, swap_entry, swapout_sink, swapout_op, swapouts);
  CreateSwapin(sa_nodes, swap_entry, swapin_op, swapins);
  CreateUpdate(sa_nodes, swapout_sink, update_op, updates);
  CreateVariables(sa_nodes, idx, swap_entry, variables, nodeptr_to_old_nid);
  CreateModelNodes(sa_nodes, idx, new_nodes, nodeptr_to_old_nid);
  ConnectSwapout(sa_nodes, idx, swapouts, swapins, updates, variables,
                 new_nodes);
  ConnectPreSwapin(sa_nodes, idx, swapins, variables);
  ConnectAllSwapin(sa_nodes, idx, swapins, swapouts, updates, variables,
                   new_nodes);
  ConnectUpdate(sa_nodes, idx, updates, swapins, variables, new_nodes);
  ConnectModelNodes(sa_nodes, idx, new_nodes, swapouts, swapins, variables);

  // Create a new graph
  std::cout << "Create a new graph" << std::endl;
  Graph ret;
  size_t output_idx = 0;
  size_t num_forward_outputs = src.GetAttr<size_t>("num_forward_outputs");
  for (const NodeEntry& e : src.outputs) {
    CHECK(new_nodes.count(idx.node_id(e.node.get())) == 1);
    ret.outputs.emplace_back(NodeEntry{new_nodes[idx.node_id(e.node.get())],
                                       e.index,
                                       e.version});
    if (output_idx + 1 == num_forward_outputs) {
      ret.outputs.emplace_back(swapout_sink);
    }
    output_idx += 1;
  }

  // Update graph attributes.
  const auto& new_idx = ret.indexed_graph();
  IdMapping new_to_old_nids;
  IdMapping old_to_new_nids;
  IdMapping new_to_old_eids;
  //std::cout << "nodeptr_to_old_nid " << nodeptr_to_old_nid.size() << std::endl;
  for (uint32_t nid = 0; nid < new_idx.num_nodes(); ++nid) {
    const auto it =
      nodeptr_to_old_nid.find(const_cast<Node*>(new_idx[nid].source));
    if (it == nodeptr_to_old_nid.end()) {
      //std::cout << "SKIP " << nid << new_idx[nid].source->attrs.name
                //<< std::endl;
      continue;
    }
    const size_t old_nid = it->second;
    new_to_old_nids[nid] = old_nid;
    old_to_new_nids[old_nid] = nid;
    const size_t num_outputs = new_idx[nid].source->num_outputs();
    for (size_t output_idx = 0; output_idx < num_outputs; output_idx++) {
      new_to_old_eids[new_idx.entry_id(nid, output_idx)] =
        idx.entry_id(old_nid, output_idx);
    }
  }
  ret.attrs["context"] = src.attrs.at("context");
  ret.attrs["device"] = src.attrs.at("device");
  ret.attrs["old_hdl_usages"] = std::make_shared<dmlc::any>(std::move(handle_usages));
  ret.attrs["hdl_sizes"] = std::make_shared<dmlc::any>(std::move(handle_sizes));
  ret.attrs["new_to_old_nids"] = std::make_shared<dmlc::any>(std::move(new_to_old_nids));
  ret.attrs["old_to_new_nids"] = std::make_shared<dmlc::any>(std::move(old_to_new_nids));
  ret.attrs["new_to_old_eids"] = std::make_shared<dmlc::any>(std::move(new_to_old_eids));
  ret.attrs["num_forward_inputs"] =
    std::make_shared<dmlc::any>(src.GetAttr<size_t>("num_forward_inputs"));
  ret.attrs["num_forward_outputs"] =
    std::make_shared<dmlc::any>(num_forward_outputs + 1);

  return ret;
}

NNVM_REGISTER_PASS(SA_LoadGraph)
.describe("Load the new dataflow graph generated by the algorithms.")
.set_body(SA_LoadGraph)
.set_change_graph(true)
.provide_graph_attr("num_forward_inputs")
.provide_graph_attr("num_forward_outputs")
.provide_graph_attr("new_to_old_nids")
.provide_graph_attr("old_to_new_nids")
.provide_graph_attr("new_to_old_eids")
.provide_graph_attr("old_hdl_usages")
.provide_graph_attr("hdl_sizes")
.depend_graph_attr("context")
.depend_graph_attr("device")
.depend_graph_attr("num_forward_inputs")
.depend_graph_attr("num_forward_outputs")
.depend_graph_attr("update_op")
.depend_graph_attr("swap_entry_op")
.depend_graph_attr("swapout_sink_op")
.depend_graph_attr("swapin_op")
.depend_graph_attr("swapout_op");
}  // namespace
}  // namespace pass
}  // namespace nnvm
